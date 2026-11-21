"""Exporte les visuels et données du poster et du portfolio, à partir des résultats réels.

Usage (depuis la racine du dépôt, après les notebooks 03 et 04) :
    python -m poster.generer_assets --sortie <dossier du site> --url https://lamiaeandof.github.io

Produit dans <sortie>/projet/ :
    xray.jpg, gradcam.png           ouverture 3D : radiographie + carte Grad-CAM réelle (RGBA)
    cas/<type>_radio.jpg, cas/<type>_cam.jpg   les 4 cas (VP, VN, FP, FN) : radio seule et superposée
    donnees_test.json               probabilités réelles du test (pour le curseur de seuil et la ROC)
et poster/qr.svg (QR code vers le portfolio).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qrcode
import qrcode.image.svg
from PIL import Image

from src.data import DOSSIER_RESULTATS, RACINE, charger_image
from src.gradcam import agrandir, selectionner_cas

TAILLE = 448


def couleurs_carte(carte: np.ndarray, alpha_max: float) -> np.ndarray:
    """Même rendu que src/gradcam.py (superposer) : couleurs chaudes, transparent sous 20 %."""
    r = np.clip((np.clip(carte, 0, 1) - 0.2) / 0.8, 0, 1)
    rgba = plt.get_cmap("inferno")(0.5 + 0.5 * r)
    rgba[..., 3] = alpha_max * r ** 0.8
    return rgba


def superposition(gris: np.ndarray, carte: np.ndarray) -> np.ndarray:
    rgba = couleurs_carte(carte, 0.75)
    fond = np.repeat(gris[..., None] / 255.0, 3, axis=2)
    a = rgba[..., 3:4]
    return ((fond * (1 - a) + rgba[..., :3] * a) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--url", required=True)
    args = ap.parse_args()
    sortie = Path(args.sortie) / "projet"
    (sortie / "cas").mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(DOSSIER_RESULTATS / "predictions_test.csv")
    cartes = np.load(DOSSIER_RESULTATS / "cartes_gradcam_test_densenet121.npy").astype(np.float32)
    metriques = json.loads((DOSSIER_RESULTATS / "metriques.json").read_text(encoding="utf-8"))

    # 1) Les 4 cas de la figure gradcam_4_cas.png (même sélection)
    cas = selectionner_cas(pred, "densenet121", 0.5, n=1)
    infos_cas = {}
    for cle, sel in cas.items():
        i = int(sel.index[0])
        gris = charger_image(pred.chemin[i], TAILLE)
        carte = agrandir(cartes[i], TAILLE)
        Image.fromarray(gris).save(sortie / "cas" / f"{cle}_radio.jpg", quality=85)
        Image.fromarray(superposition(gris, carte)).save(sortie / "cas" / f"{cle}_cam.jpg", quality=85)
        infos_cas[cle] = {"fichier": pred.fichier[i], "proba": round(float(pred.proba_densenet121[i]), 3)}

    # 2) Ouverture 3D : le vrai positif de la galerie
    i = int(cas["VP"].index[0])
    Image.fromarray(charger_image(pred.chemin[i], 512)).save(sortie / "xray.jpg", quality=86)
    rgba = couleurs_carte(agrandir(cartes[i], 512), 0.85)
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(sortie / "gradcam.png", optimize=True)

    # 3) Données réelles du test pour les graphiques interactifs
    donnees = {
        "label": pred.label.astype(int).tolist(),
        "densenet121": [round(float(p), 5) for p in pred.proba_densenet121],
        "baseline": [round(float(p), 5) for p in pred.proba_baseline],
        "seuils_haute_sensibilite": {m: metriques[m]["haute_sensibilite"]["test"]["seuil"] for m in ["densenet121", "baseline"]},
        "auc": {m: metriques[m]["0.5"]["test"]["auc"] for m in ["densenet121", "baseline"]},
        "cas": infos_cas,
    }
    (sortie / "donnees_test.json").write_text(json.dumps(donnees, separators=(",", ":")), encoding="utf-8")

    # 4) QR code vers le portfolio
    qr = qrcode.make(args.url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=1)
    qr.save(str(RACINE / "poster" / "qr.svg"))
    print("cas :", infos_cas)


if __name__ == "__main__":
    main()
