"""Évaluation sur le test officiel : métriques médicales, choix du seuil, bootstrap, figures.

Usage :
    python -m src.evaluate

Sorties :
    results/predictions_val.csv, results/predictions_test.csv   probabilités par image
    results/metriques.json                                      toutes les métriques + IC 95 %
    results/tableau_resultats.md                                tableau prêt pour le README
    figures/roc_test.png, figures/matrices_confusion.png, figures/distribution_probas.png
"""
from __future__ import annotations

import json
import sys
import math

import matplotlib

if "ipykernel" not in sys.modules:  # scripts : pas de fenêtre ; notebooks : affichage normal
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.data import (DOSSIER_FIGURES, DOSSIER_MODELES, DOSSIER_RESULTATS,
                      charger_split_et_cache, dataset_pour)
from src.models import charger_modele
from src.train import predire

MODELES = {"baseline": "Baseline (petit CNN)", "densenet121": "DenseNet-121"}
COULEURS = {"baseline": "#eb6834", "densenet121": "#2a78d6"}  # palette catégorielle validée
MARINE = "#14284b"     # titres / texte
GRIS = "#52514e"       # texte secondaire
CIBLE_SENSIBILITE = 0.98  # fixée À L'AVANCE : au plus 2 % de pneumonies manquées (sur la validation)


# --------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------
def metriques_au_seuil(y: np.ndarray, p: np.ndarray, seuil: float) -> dict:
    """Matrice de confusion et métriques médicales (pneumonie = classe positive)."""
    pred = p >= seuil
    vp = int(np.sum(pred & (y == 1)))   # pneumonies détectées
    fn = int(np.sum(~pred & (y == 1)))  # pneumonies ratées (le plus grave)
    vn = int(np.sum(~pred & (y == 0)))  # normales reconnues
    fp = int(np.sum(pred & (y == 0)))   # fausses alertes
    div = lambda a, b: a / b if b else float("nan")
    sens, spec, prec = div(vp, vp + fn), div(vn, vn + fp), div(vp, vp + fp)
    return {
        "seuil": float(seuil), "VP": vp, "FN": fn, "VN": vn, "FP": fp,
        "sensibilite": sens, "specificite": spec, "precision": prec,
        "vpn": div(vn, vn + fn),
        "f1": div(2 * prec * sens, prec + sens),
        "accuracy": div(vp + vn, len(y)),
        "auc": float(roc_auc_score(y, p)),
    }


def seuil_pour_sensibilite(y_val: np.ndarray, p_val: np.ndarray, cible: float = CIBLE_SENSIBILITE) -> float:
    """Plus grand seuil qui atteint la sensibilité cible SUR LA VALIDATION.

    Le seuil est choisi sur la validation puis appliqué tel quel au test :
    choisir le seuil sur le test reviendrait à « tricher » (le test doit rester inédit).
    """
    p_pos = np.sort(p_val[y_val == 1])[::-1]       # probabilités des pneumonies, décroissantes
    k = math.ceil(cible * len(p_pos))              # nombre de pneumonies à détecter au minimum
    return float(p_pos[k - 1])


def bootstrap_ic(y: np.ndarray, p: np.ndarray, seuil: float, n: int = 2000, seed: int = 42) -> dict:
    """Intervalles de confiance à 95 % par bootstrap (rééchantillonnage des images du test).

    On tire n fois 624 images avec remise, on recalcule les métriques, et on garde
    les percentiles 2,5 % et 97,5 %. Cela montre l'incertitude due à la taille du test.
    """
    rng = np.random.default_rng(seed)
    cles = ["sensibilite", "specificite", "precision", "f1", "accuracy", "auc"]
    tirages = {c: [] for c in cles}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = metriques_au_seuil(y[idx], p[idx], seuil)
        for c in cles:
            tirages[c].append(m[c])
    return {c: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for c, v in tirages.items()}


# --------------------------------------------------------------------------
# Prédictions
# --------------------------------------------------------------------------
def calculer_predictions(device=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probabilités de pneumonie des deux modèles sur la validation et le test."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, cache = charger_split_et_cache()
    sorties = {}
    for split in ["val", "test"]:
        ds = dataset_pour(split, df, cache, entrainement=False)
        dl = DataLoader(ds, batch_size=32, shuffle=False)
        res = df[df["split"] == split][["chemin", "fichier", "patient", "sous_type", "label"]].copy()
        for nom in MODELES:
            modele = charger_modele(nom, DOSSIER_MODELES / f"{nom}_meilleur.pt", device)
            res[f"proba_{nom}"], _, _ = predire(modele, dl, device)
        sorties[split] = res.reset_index(drop=True)
    return sorties["val"], sorties["test"]


def evaluer_tout(pred_val: pd.DataFrame, pred_test: pd.DataFrame) -> dict:
    """Métriques au seuil 0,5 et au seuil « haute sensibilité » (choisi sur la validation)."""
    resultats = {}
    for nom in MODELES:
        y_val, p_val = pred_val["label"].to_numpy(), pred_val[f"proba_{nom}"].to_numpy()
        y, p = pred_test["label"].to_numpy(), pred_test[f"proba_{nom}"].to_numpy()
        seuils = {"0.5": 0.5, "haute_sensibilite": seuil_pour_sensibilite(y_val, p_val)}
        resultats[nom] = {"val_auc": float(roc_auc_score(y_val, p_val))}
        for cle, s in seuils.items():
            resultats[nom][cle] = {
                "val": metriques_au_seuil(y_val, p_val, s),
                "test": metriques_au_seuil(y, p, s),
                "test_ic95": bootstrap_ic(y, p, s),
            }
    return resultats


def tableau_markdown(resultats: dict) -> str:
    """Tableau des métriques sur le test (avec IC 95 %)."""
    f = lambda v, ic: f"{v:.3f} [{ic[0]:.3f}–{ic[1]:.3f}]"
    lignes = ["| Modèle | Seuil | AUC-ROC | Sensibilité | Spécificité | Précision | F1 | Accuracy | FN | FP |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for nom, libelle in MODELES.items():
        for cle in ["0.5", "haute_sensibilite"]:
            m, ic = resultats[nom][cle]["test"], resultats[nom][cle]["test_ic95"]
            seuil = "0,50" if cle == "0.5" else f"{m['seuil']:.3f} (sens. val ≥ {CIBLE_SENSIBILITE:.0%})"
            lignes.append(
                f"| {libelle} | {seuil} | {f(m['auc'], ic['auc'])} | {f(m['sensibilite'], ic['sensibilite'])} | "
                f"{f(m['specificite'], ic['specificite'])} | {f(m['precision'], ic['precision'])} | "
                f"{f(m['f1'], ic['f1'])} | {f(m['accuracy'], ic['accuracy'])} | {m['FN']} | {m['FP']} |")
    return "\n".join(lignes)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style(ax):
    """Style sobre : axes discrets, grille légère, texte en encre neutre."""
    for cote in ["top", "right"]:
        ax.spines[cote].set_visible(False)
    for cote in ["left", "bottom"]:
        ax.spines[cote].set_color("#b9b8b1")
    ax.tick_params(colors=GRIS, labelsize=9)
    ax.grid(color="#e7e6e1", linewidth=0.8)
    ax.set_axisbelow(True)


def figure_roc(pred_test: pd.DataFrame, resultats: dict, chemin=None):
    """Courbes ROC des deux modèles sur le test, avec les deux points de fonctionnement."""
    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=150)
    ax.plot([0, 1], [0, 1], color="#b9b8b1", lw=1, ls="--")
    y = pred_test["label"].to_numpy()
    for nom, libelle in MODELES.items():
        p = pred_test[f"proba_{nom}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        auc = resultats[nom]["0.5"]["test"]["auc"]
        ax.plot(fpr, tpr, color=COULEURS[nom], lw=2, label=f"{libelle} — AUC {auc:.3f}")
        for cle, marque in [("0.5", "o"), ("haute_sensibilite", "s")]:
            m = resultats[nom][cle]["test"]
            ax.scatter(1 - m["specificite"], m["sensibilite"], s=55, marker=marque,
                       color=COULEURS[nom], edgecolor="white", linewidth=1.5, zorder=3)
    # Légende des marqueurs (forme = seuil), indépendante de la couleur
    ax.scatter([], [], marker="o", color=GRIS, label="seuil 0,50")
    ax.scatter([], [], marker="s", color=GRIS, label=f"seuil haute sensibilité")
    ax.set_xlabel("1 − spécificité (taux de fausses alertes)", color=MARINE)
    ax.set_ylabel("Sensibilité (pneumonies détectées)", color=MARINE)
    ax.set_title("Courbe ROC — test officiel (624 images)", color=MARINE, fontsize=11, loc="left")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    _style(ax)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=MARINE)
    fig.tight_layout()
    if chemin:
        fig.savefig(chemin, bbox_inches="tight")
    return fig


def figure_matrices(resultats: dict, nom: str = "densenet121", chemin=None):
    """Matrices de confusion du modèle principal aux deux seuils."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=150)
    for ax, cle in zip(axes, ["0.5", "haute_sensibilite"]):
        m = resultats[nom][cle]["test"]
        mat = np.array([[m["VN"], m["FP"]], [m["FN"], m["VP"]]])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.sum())
        for i in range(2):
            for j in range(2):
                pourc = mat[i, j] / mat[i].sum()
                ax.text(j, i, f"{mat[i, j]}\n({pourc:.0%} de la ligne)", ha="center", va="center",
                        fontsize=10, color="white" if mat[i, j] > mat.sum() * 0.45 else MARINE)
        ax.set_xticks([0, 1], ["prédit NORMAL", "prédit PNEUMONIE"], fontsize=9, color=GRIS)
        ax.set_yticks([0, 1], ["vrai NORMAL", "vrai PNEUMONIE"], fontsize=9, color=GRIS)
        titre = "Seuil 0,50" if cle == "0.5" else f"Seuil {m['seuil']:.3f} (haute sensibilité)"
        ax.set_title(f"{titre}\nsens. {m['sensibilite']:.3f} · spéc. {m['specificite']:.3f}",
                     color=MARINE, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"{MODELES[nom]} — matrices de confusion sur le test", color=MARINE, fontsize=11)
    fig.tight_layout()
    if chemin:
        fig.savefig(chemin, bbox_inches="tight")
    return fig


def figure_distribution(pred_val: pd.DataFrame, pred_test: pd.DataFrame, resultats: dict,
                        nom: str = "densenet121", chemin=None):
    """Distribution des probabilités prédites par vraie classe, validation vs test.

    Permet de voir le décalage entre la validation (issue du train) et le test officiel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150, sharey=False)
    bins = np.linspace(0, 1, 26)
    for ax, (titre, d) in zip(axes, [("Validation", pred_val), ("Test officiel", pred_test)]):
        for label, libelle, couleur in [(0, "vrai NORMAL", "#1baf7a"), (1, "vrai PNEUMONIE", "#6250d6")]:
            p = d.loc[d["label"] == label, f"proba_{nom}"]
            ax.hist(p, bins=bins, color=couleur, alpha=0.75, label=f"{libelle} (n={len(p)})",
                    edgecolor="white", linewidth=1)
        for cle, style in [("0.5", "-"), ("haute_sensibilite", "--")]:
            s = resultats[nom][cle]["test"]["seuil"]
            ax.axvline(s, color=MARINE, ls=style, lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Probabilité de pneumonie prédite", color=MARINE, fontsize=9)
        ax.set_title(titre, color=MARINE, fontsize=10, loc="left")
        _style(ax)
        ax.legend(fontsize=8, frameon=False, labelcolor=MARINE, loc="upper center")
    axes[0].set_ylabel("Nombre d'images (échelle log)", color=MARINE, fontsize=9)
    fig.suptitle(f"{MODELES[nom]} — trait plein : seuil 0,50 ; pointillés : seuil haute sensibilité",
                 color=GRIS, fontsize=9)
    fig.tight_layout()
    if chemin:
        fig.savefig(chemin, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
def main():
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    pred_val, pred_test = calculer_predictions()
    pred_val.to_csv(DOSSIER_RESULTATS / "predictions_val.csv", index=False)
    pred_test.to_csv(DOSSIER_RESULTATS / "predictions_test.csv", index=False)

    resultats = evaluer_tout(pred_val, pred_test)
    with open(DOSSIER_RESULTATS / "metriques.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    tableau = tableau_markdown(resultats)
    (DOSSIER_RESULTATS / "tableau_resultats.md").write_text(tableau, encoding="utf-8")
    print(tableau)

    figure_roc(pred_test, resultats, DOSSIER_FIGURES / "roc_test.png")
    figure_matrices(resultats, chemin=DOSSIER_FIGURES / "matrices_confusion.png")
    figure_distribution(pred_val, pred_test, resultats, chemin=DOSSIER_FIGURES / "distribution_probas.png")


if __name__ == "__main__":
    main()
