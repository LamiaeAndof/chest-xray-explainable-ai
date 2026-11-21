"""Démo pédagogique : on dépose une radiographie, on obtient P(pneumonie) et la carte Grad-CAM.

DÉMONSTRATION PÉDAGOGIQUE — PAS UN OUTIL DE DIAGNOSTIC.

Lancement (depuis la racine du dépôt, après l'entraînement) :
    pip install gradio
    python -m app.demo_gradio
"""
from __future__ import annotations

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data import DOSSIER_MODELES, TAILLE, redimensionner_recadrer
from src.gradcam import cartes_gradcam, preparer_entree
from src.models import charger_modele

AVERTISSEMENT = ("⚠️ Démonstration pédagogique — pas un outil de diagnostic. Modèle entraîné sur des "
                 "radiographies pédiatriques d'un seul hôpital ; aucune validation clinique.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modele = charger_modele("densenet121", DOSSIER_MODELES / "densenet121_meilleur.pt", device)


def analyser(image: Image.Image):
    if image is None:
        return None, None
    gris = redimensionner_recadrer(image, TAILLE)   # même prétraitement qu'à l'entraînement
    # Probabilité de pneumonie
    with torch.no_grad():
        p = torch.sigmoid(modele(preparer_entree(gris[None]).to(device))).item()
    classe = int(p >= 0.5)
    carte = cartes_gradcam(modele, gris[None], [classe])[0]
    # Superposition (même rendu que les figures du projet)
    couleurs = plt.get_cmap("inferno")(carte)[..., :3]
    alpha = (carte * 0.65)[..., None]
    fond = np.repeat(gris[..., None] / 255.0, 3, axis=2)
    superpose = (fond * (1 - alpha) + couleurs * alpha)
    return {"PNEUMONIE": p, "NORMAL": 1 - p}, (superpose * 255).astype(np.uint8)


demo = gr.Interface(
    fn=analyser,
    inputs=gr.Image(type="pil", label="Radiographie thoracique (face)"),
    outputs=[gr.Label(label="Probabilités (seuil 0,5)"),
             gr.Image(label="Grad-CAM : zones qui ont poussé vers la classe prédite")],
    title="IA explicable — pneumonie sur radiographie thoracique (DenseNet-121 + Grad-CAM)",
    description=AVERTISSEMENT,
    article=AVERTISSEMENT,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
