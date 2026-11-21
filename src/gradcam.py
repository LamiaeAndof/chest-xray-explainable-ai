"""Explicabilité : cartes Grad-CAM (Selvaraju et al., 2017) avec la bibliothèque
pytorch-grad-cam (J. Gildenblat).

Principe de Grad-CAM : on prend les cartes d'activation de la dernière couche
convolutive (7x7 pour DenseNet-121), on les pondère par la moyenne du gradient du
score de la classe expliquée, on somme et on garde la partie positive (ReLU).
Résultat : une carte basse résolution, agrandie à la taille de l'image, qui montre
les zones qui ont fait MONTER le score de cette classe.

Usage :
    python -m src.gradcam      (figures dans figures/, statistiques dans results/)
"""
from __future__ import annotations

import json
import sys

import matplotlib

if "ipykernel" not in sys.modules:  # scripts : pas de fenêtre ; notebooks : affichage normal
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from src.data import (DOSSIER_FIGURES, DOSSIER_MODELES, DOSSIER_RESULTATS, TAILLE,
                      charger_image, charger_split_et_cache, transformations)
from src.models import charger_modele, couche_cible_gradcam

METHODES = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus}
MARINE, GRIS = "#14284b", "#52514e"
LIBELLES_CAS = {
    "VP": "Vrai positif (pneumonie détectée)",
    "VN": "Vrai négatif (normale reconnue)",
    "FP": "Faux positif (fausse alerte)",
    "FN": "Faux négatif (pneumonie ratée)",
}


def preparer_entree(images_uint8: np.ndarray) -> torch.Tensor:
    """(N, H, W) uint8 -> tenseur normalisé (N, 3, H, W), sans augmentation."""
    t = transformations(entrainement=False)
    return torch.stack([t(torch.from_numpy(im).unsqueeze(0)) for im in images_uint8])


def cartes_gradcam(modele, images_uint8: np.ndarray, classes, methode: str = "gradcam",
                   batch: int = 2) -> np.ndarray:
    """Cartes (N, H, W) dans [0, 1] pour la classe demandée (1 = pneumonie, 0 = normale).

    BinaryClassifierOutputTarget(1) explique le logit (évidence de pneumonie) ;
    BinaryClassifierOutputTarget(0) explique -logit (évidence de radio normale).
    """
    device = next(modele.parameters()).device
    cartes = []
    with METHODES[methode](model=modele, target_layers=[couche_cible_gradcam(modele)]) as cam:
        for i in range(0, len(images_uint8), batch):
            x = preparer_entree(images_uint8[i:i + batch]).to(device)
            cibles = [BinaryClassifierOutputTarget(int(c)) for c in classes[i:i + batch]]
            cartes.append(cam(input_tensor=x, targets=cibles))
    return np.concatenate(cartes)


def part_energie_bord(cartes: np.ndarray, marge: float = 0.10) -> np.ndarray:
    """Part de la « masse » de chaque carte située dans une bande de bord de l'image.

    Bande = les 10 % extérieurs de chaque côté (36 % de la surface). Une carte
    uniforme donnerait 0,36. Une valeur élevée signale un modèle qui s'appuie sur
    les bords (marqueurs, texte, épaules...) plutôt que sur les poumons.
    """
    n, h, w = cartes.shape
    mh, mw = int(round(h * marge)), int(round(w * marge))
    centre = cartes[:, mh:h - mh, mw:w - mw].sum(axis=(1, 2))
    total = cartes.sum(axis=(1, 2)) + 1e-8
    return 1 - centre / total


def selectionner_cas(pred: pd.DataFrame, nom: str, seuil: float, n: int = 1) -> dict:
    """Pour chaque catégorie (VP, VN, FP, FN), les n cas où le modèle est le plus sûr de lui.

    Les erreurs commises avec une grande confiance sont les plus instructives.
    """
    p, y = pred[f"proba_{nom}"], pred["label"]
    groupes = {
        "VP": pred[(y == 1) & (p >= seuil)].sort_values(f"proba_{nom}", ascending=False),
        "VN": pred[(y == 0) & (p < seuil)].sort_values(f"proba_{nom}", ascending=True),
        "FP": pred[(y == 0) & (p >= seuil)].sort_values(f"proba_{nom}", ascending=False),
        "FN": pred[(y == 1) & (p < seuil)].sort_values(f"proba_{nom}", ascending=True),
    }
    return {k: g.head(n) for k, g in groupes.items()}


def superposer(ax, image_gris: np.ndarray, carte: np.ndarray):
    """Radiographie en gris + carte en couleur, transparente là où la carte est faible."""
    ax.imshow(image_gris, cmap="gray", vmin=0, vmax=255)
    # Opacité nulle sous 20 % de la carte, puis croissante jusqu'à 0,75 au maximum ;
    # couleur prise dans la partie chaude de la palette (orange -> jaune).
    # Ainsi, seules les zones réellement utilisées par le modèle sont colorées.
    r = np.clip((np.clip(carte, 0, 1) - 0.2) / 0.8, 0, 1)
    couleurs = plt.get_cmap("inferno")(0.5 + 0.5 * r)
    couleurs[..., 3] = 0.75 * r ** 0.8
    ax.imshow(couleurs)


def agrandir(carte: np.ndarray, taille: int) -> np.ndarray:
    """Agrandit une carte 224x224 à la taille d'affichage (interpolation bilinéaire)."""
    t = torch.from_numpy(carte)[None, None].float()
    return F.interpolate(t, size=(taille, taille), mode="bilinear", align_corners=False)[0, 0].numpy()


def figure_cas(lignes: list[dict], chemin=None, taille_affichage: int = 448):
    """Une ligne par cas : radiographie | carte Grad-CAM | superposition.

    Chaque dict contient : chemin, titre, sous_titre, carte (224x224).
    """
    fig, axes = plt.subplots(len(lignes), 3, figsize=(9, 3.15 * len(lignes)), dpi=150)
    axes = np.atleast_2d(axes)
    for ax_ligne, cas in zip(axes, lignes):
        img = charger_image(cas["chemin"], taille_affichage)
        carte = agrandir(cas["carte"], taille_affichage)
        ax_ligne[0].imshow(img, cmap="gray", vmin=0, vmax=255)
        ax_ligne[1].imshow(carte, cmap="inferno", vmin=0, vmax=1)
        superposer(ax_ligne[2], img, carte)
        ax_ligne[0].set_title(cas["titre"], loc="left", fontsize=10, color=MARINE, fontweight="bold")
        ax_ligne[1].set_title("Carte Grad-CAM", fontsize=9, color=GRIS)
        ax_ligne[2].set_title(cas["sous_titre"], fontsize=9, color=GRIS)
        for ax in ax_ligne:
            ax.axis("off")
    fig.tight_layout()
    if chemin:
        fig.savefig(chemin, bbox_inches="tight")
    return fig


def figure_carte_moyenne(cartes: np.ndarray, pred_classes: np.ndarray, image_moyenne: np.ndarray,
                         marge: float = 0.10, chemin=None):
    """Carte Grad-CAM moyenne sur tout le test, par classe prédite, avec la bande de bord."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), dpi=150)
    for ax, c, titre in zip(axes, [1, 0], ["Prédites PNEUMONIE", "Prédites NORMALES"]):
        sel = cartes[pred_classes == c]
        moyenne = sel.mean(axis=0)
        superposer(ax, image_moyenne, moyenne / (moyenne.max() + 1e-8))
        h = cartes.shape[1]
        m = h * marge
        ax.add_patch(plt.Rectangle((m, m), h - 2 * m, h - 2 * m, fill=False, ec="white", ls="--", lw=1))
        ax.set_title(f"{titre} (n={len(sel)})", fontsize=10, color=MARINE)
        ax.axis("off")
    fig.suptitle("Carte Grad-CAM moyenne sur le test (fond : radiographie moyenne ;\n"
                 "pointillés : limite de la bande de bord de 10 %)", fontsize=9, color=GRIS)
    fig.tight_layout()
    if chemin:
        fig.savefig(chemin, bbox_inches="tight")
    return fig


def main(nom: str = "densenet121"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, cache = charger_split_et_cache()
    pred = pd.read_csv(DOSSIER_RESULTATS / "predictions_test.csv")
    seuil = 0.5
    modele = charger_modele(nom, DOSSIER_MODELES / f"{nom}_meilleur.pt", device)

    # Images du test dans le même ordre que predictions_test.csv
    images_test = cache[(df["split"] == "test").to_numpy()]
    assert len(images_test) == len(pred)
    pred_classes = (pred[f"proba_{nom}"] >= seuil).astype(int).to_numpy()

    # 1) Cartes pour TOUT le test, en expliquant la classe prédite
    cartes = cartes_gradcam(modele, images_test, pred_classes)
    np.save(DOSSIER_RESULTATS / f"cartes_gradcam_test_{nom}.npy", cartes.astype(np.float16))

    # 2) Statistique « bords » : le modèle regarde-t-il les poumons ou les bords ?
    bord = part_energie_bord(cartes)
    pred["part_bord"] = bord
    stats = {
        "seuil": seuil, "marge": 0.10, "part_surface_bord": 0.36,
        "part_bord_moyenne": float(bord.mean()), "part_bord_mediane": float(np.median(bord)),
        "part_bord_moyenne_par_classe_predite": {
            "PNEUMONIA": float(bord[pred_classes == 1].mean()),
            "NORMAL": float(bord[pred_classes == 0].mean())},
        "n_cartes_majoritairement_bord(>0.5)": int((bord > 0.5).sum()),
    }
    with open(DOSSIER_RESULTATS / f"gradcam_stats_{nom}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 3) Figure des 4 types de cas
    cas = selectionner_cas(pred, nom, seuil, n=1)
    lignes = []
    for cle, sel in cas.items():
        for i, r in sel.iterrows():
            lignes.append({"chemin": r["chemin"], "carte": cartes[i], "titre": LIBELLES_CAS[cle],
                           "sous_titre": f"P(pneumonie) = {r[f'proba_{nom}']:.3f} · part bord {bord[i]:.2f}"})
    figure_cas(lignes, DOSSIER_FIGURES / "gradcam_4_cas.png")

    # 4) Carte moyenne
    figure_carte_moyenne(cartes, pred_classes, images_test.mean(axis=0),
                         chemin=DOSSIER_FIGURES / "gradcam_carte_moyenne.png")


if __name__ == "__main__":
    main()
