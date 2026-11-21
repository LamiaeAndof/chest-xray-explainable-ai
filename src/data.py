"""Données : listing des images, identifiants patients, découpage par patient,
cache des images redimensionnées et transformations.

Usage en ligne de commande (prépare tout ce dont l'entraînement a besoin) :
    python -m src.data
"""
from __future__ import annotations

import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import v2

# --------------------------------------------------------------------------
# Chemins du projet (tout est relatif à la racine du dépôt)
# --------------------------------------------------------------------------
RACINE = Path(__file__).resolve().parents[1]
DOSSIER_BRUT = RACINE / "data" / "raw"
DOSSIER_DONNEES = DOSSIER_BRUT / "chest_xray"
DOSSIER_CACHE = RACINE / "data" / "cache"
DOSSIER_RESULTATS = RACINE / "results"
DOSSIER_FIGURES = RACINE / "figures"
DOSSIER_MODELES = RACINE / "models"

TAILLE = 224  # taille d'entrée standard des réseaux pré-entraînés sur ImageNet
SEED = 42  # graine fixe pour la reproductibilité
CLASSES = ["NORMAL", "PNEUMONIA"]  # label 0 = NORMAL, label 1 = PNEUMONIA

# Normalisation ImageNet : le DenseNet a été pré-entraîné avec ces statistiques
MOYENNE_IMAGENET = [0.485, 0.456, 0.406]
ECART_TYPE_IMAGENET = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# 1. Extraction de l'archive Kaggle
# --------------------------------------------------------------------------
def extraire_archive(zip_path: Path = DOSSIER_BRUT / "chest-xray-pneumonia.zip") -> None:
    """Extrait uniquement chest_xray/{train,val,test}.

    L'archive Kaggle contient aussi une copie en double (chest_xray/chest_xray/)
    et un dossier __MACOSX : on les ignore pour ne pas compter deux fois les images.
    """
    if (DOSSIER_DONNEES / "train").exists():
        return
    prefixes = ("chest_xray/train/", "chest_xray/val/", "chest_xray/test/")
    with zipfile.ZipFile(zip_path) as zf:
        membres = [m for m in zf.namelist() if m.startswith(prefixes)]
        zf.extractall(DOSSIER_BRUT, members=membres)


# --------------------------------------------------------------------------
# 2. Listing des images et identifiants patients
# --------------------------------------------------------------------------
# Pneumonies : person1234_bacteria_5678.jpeg, person1234_virus_5678.jpeg
#              (et quelques variantes : person447_virus_921_1.jpeg)
MOTIF_PNEUMONIE = re.compile(r"^(person\d+)_(bacteria|virus)_\d+(?:_\d+)*\.jpe?g$", re.IGNORECASE)
# Normales : IM-0115-0001.jpeg, NORMAL2-IM-1427-0001.jpeg
#            (et quelques variantes : IM-0429-0001-0002.jpeg)
MOTIF_NORMAL = re.compile(r"^(?:NORMAL2-)?(IM-\d+)(?:-\d+)+\.jpe?g$", re.IGNORECASE)


def extraire_patient(nom_fichier: str) -> tuple[str, str]:
    """Renvoie (groupe_patient, sous_type) à partir du nom de fichier.

    sous_type vaut 'bacteria', 'virus' ou 'normal'.
    Pour les normales, « IM-0651 » et « NORMAL2-IM-0651 » sont mis dans le MÊME groupe :
    on ne sait pas s'il s'agit du même enfant, et par prudence on les regroupe
    (regrouper à tort ne crée jamais de fuite ; séparer à tort pourrait en créer).
    Si le nom ne suit aucun motif connu, le fichier forme un groupe à lui seul.
    """
    m = MOTIF_PNEUMONIE.match(nom_fichier)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = MOTIF_NORMAL.match(nom_fichier)
    if m:
        return m.group(1).upper(), "normal"
    return nom_fichier, "inconnu"


def lister_images(dossier: Path = DOSSIER_DONNEES) -> pd.DataFrame:
    """Une ligne par image : split officiel, classe, label, patient, sous-type."""
    lignes = []
    for split in ["train", "val", "test"]:
        for classe in CLASSES:
            for chemin in sorted((dossier / split / classe).glob("*.jp*g")):
                patient, sous_type = extraire_patient(chemin.name)
                lignes.append({
                    "chemin": chemin.relative_to(DOSSIER_DONNEES).as_posix(),
                    "fichier": chemin.name,
                    "split_officiel": split,
                    "classe": classe,
                    "label": CLASSES.index(classe),
                    "patient": patient,
                    "sous_type": sous_type,
                })
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------
# 3. Découpage train / validation PAR PATIENT
# --------------------------------------------------------------------------
def decoupage_par_patient(df: pd.DataFrame, n_plis: int = 7, seed: int = SEED) -> pd.DataFrame:
    """Ajoute une colonne 'split' : 'train', 'val' ou 'test'.

    - Le test officiel reste intact (il ne sert qu'à l'évaluation finale).
    - Le train officiel + le val officiel (16 images, trop petit) forment un pool
      que l'on redécoupe : 1 pli sur 7 (~14 %) devient la validation.
    - StratifiedGroupKFold garantit (1) qu'un même patient n'est jamais à la fois
      en train et en validation (groupes = patients) et (2) que la proportion de
      pneumonies est à peu près la même dans les deux ensembles (stratification).
    """
    df = df.copy()
    df["split"] = np.where(df["split_officiel"] == "test", "test", "train")
    pool = df[df["split"] == "train"]
    sgkf = StratifiedGroupKFold(n_splits=n_plis, shuffle=True, random_state=seed)
    _, idx_val = next(sgkf.split(pool, y=pool["label"], groups=pool["patient"]))
    df.loc[pool.index[idx_val], "split"] = "val"
    return df


def verifier_fuite(df: pd.DataFrame) -> dict:
    """Compte les patients présents dans plusieurs ensembles (doit être 0 entre train et val)."""
    patients = {s: set(df.loc[df["split"] == s, "patient"]) for s in ["train", "val", "test"]}
    return {
        "train∩val": len(patients["train"] & patients["val"]),
        "train∩test": len(patients["train"] & patients["test"]),
        "val∩test": len(patients["val"] & patients["test"]),
    }


# --------------------------------------------------------------------------
# 4. Cache des images redimensionnées (accélère énormément l'entraînement)
# --------------------------------------------------------------------------
def redimensionner_recadrer(img: Image.Image, taille: int = TAILLE) -> np.ndarray:
    """Niveaux de gris, petit côté ramené à `taille`, puis recadrage central carré.

    Pourquoi pas un simple redimensionnement en carré ? Dans ce jeu de données, la
    FORME de l'image dépend de la classe (les pneumonies sont plus larges : rapport
    largeur/hauteur médian 1,49 contre 1,23 pour les normales ; ce rapport seul donne
    une AUC de 0,87 sur le train). Écraser l'image en carré transformerait cette forme
    en déformation de l'anatomie, un raccourci facile pour le modèle. Le recadrage
    central conserve les proportions réelles et masque la forme d'origine.
    """
    img = img.convert("L")  # quelques images sont en RVB : on force 1 canal
    w, h = img.size
    echelle = taille / min(w, h)
    nw, nh = max(taille, round(w * echelle)), max(taille, round(h * echelle))
    img = img.resize((nw, nh), Image.BILINEAR)
    gauche, haut = (nw - taille) // 2, (nh - taille) // 2
    img = img.crop((gauche, haut, gauche + taille, haut + taille))
    return np.asarray(img, dtype=np.uint8)


def charger_image(chemin_relatif: str, taille: int = TAILLE) -> np.ndarray:
    """Charge une radiographie du jeu de données et applique `redimensionner_recadrer`."""
    with Image.open(DOSSIER_DONNEES / chemin_relatif) as img:
        return redimensionner_recadrer(img, taille)


def construire_cache(df: pd.DataFrame, taille: int = TAILLE) -> np.ndarray:
    """Charge toutes les images une seule fois et les sauvegarde en .npy (uint8)."""
    fichier = DOSSIER_CACHE / f"images_{taille}.npy"
    if fichier.exists():
        cache = np.load(fichier)
        if len(cache) == len(df):
            return cache
    DOSSIER_CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        images = list(ex.map(lambda c: charger_image(c, taille), df["chemin"]))
    cache = np.stack(images)
    np.save(fichier, cache)
    return cache


def charger_split_et_cache() -> tuple[pd.DataFrame, np.ndarray]:
    """Charge le découpage sauvegardé (results/split.csv) et le cache d'images aligné."""
    df = pd.read_csv(DOSSIER_RESULTATS / "split.csv")
    cache = np.load(DOSSIER_CACHE / f"images_{TAILLE}.npy")
    assert len(cache) == len(df), "Cache et split désalignés : relancer `python -m src.data`"
    return df, cache


# --------------------------------------------------------------------------
# 5. Transformations et Dataset PyTorch
# --------------------------------------------------------------------------
def _trois_canaux(x: torch.Tensor) -> torch.Tensor:
    """Répète le canal gris 3 fois (fonction nommée : compatible multiprocessing Windows)."""
    return x.expand(3, -1, -1)


def transformations(entrainement: bool) -> v2.Compose:
    """Transformations appliquées à un tenseur uint8 de forme (1, H, W).

    Augmentations volontairement modestes et réalistes pour une radiographie :
    - petite rotation (±10°), légère translation (5 %) et zoom (±5 %) : variations
      normales de positionnement du patient ;
    - légère variation de luminosité / contraste : différences entre appareils.
    PAS de flip vertical (une radio n'est jamais à l'envers) et PAS de flip
    horizontal : le cœur est à gauche, un flip créerait une anatomie inversée
    (situs inversus, très rare) et ferait perdre l'information de latéralité.
    """
    etapes = []
    if entrainement:
        etapes += [
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    etapes += [
        v2.ToDtype(torch.float32, scale=True),           # uint8 [0,255] -> float [0,1]
        v2.Lambda(_trois_canaux),                        # 1 canal -> 3 canaux (format ImageNet)
        v2.Normalize(MOYENNE_IMAGENET, ECART_TYPE_IMAGENET),
    ]
    return v2.Compose(etapes)


class RadiographiesDataset(Dataset):
    """Dataset en mémoire : images uint8 (N, H, W) + labels (N,)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, entrainement: bool):
        self.images = torch.from_numpy(images)
        self.labels = torch.tensor(np.array(labels), dtype=torch.float32)  # copie (pandas 3 : lecture seule)
        self.transform = transformations(entrainement)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.transform(self.images[i].unsqueeze(0))
        return x, self.labels[i]


def dataset_pour(split: str, df: pd.DataFrame, cache: np.ndarray, entrainement: bool | None = None):
    """Construit le Dataset d'un split ('train', 'val' ou 'test')."""
    masque = (df["split"] == split).to_numpy()
    if entrainement is None:
        entrainement = split == "train"
    return RadiographiesDataset(cache[masque], df.loc[masque, "label"].to_numpy(), entrainement)


# --------------------------------------------------------------------------
# Point d'entrée : extraction, découpage, vérifications, cache
# --------------------------------------------------------------------------
if __name__ == "__main__":
    extraire_archive()
    df = lister_images()
    print(f"{len(df)} images trouvées")
    print(df.groupby(["split_officiel", "classe"]).size().unstack())
    print("Sous-types non reconnus :", (df["sous_type"] == "inconnu").sum())

    df = decoupage_par_patient(df)
    print("\nNouveau découpage (images) :")
    print(df.groupby(["split", "classe"]).size().unstack())
    print("\nPatients partagés entre ensembles :", verifier_fuite(df))

    DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DOSSIER_RESULTATS / "split.csv", index=False)

    cache = construire_cache(df)
    print(f"\nCache : {cache.shape}, {cache.nbytes / 1e6:.0f} Mo")
