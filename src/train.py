"""Entraînement d'un modèle (baseline ou DenseNet-121).

Exemples :
    python -m src.train --modele baseline --epochs 15 --lr 1e-3 --patience 4
    python -m src.train --modele densenet121 --epochs-tete 2 --epochs 8 --lr 1e-4

Sorties :
    models/<modele>_meilleur.pt          poids du meilleur modèle (AUC de validation max)
    results/historique_<modele>.json      métriques à chaque époque
    results/entrainement_<modele>.json    configuration et résumé
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.data import DOSSIER_MODELES, DOSSIER_RESULTATS, charger_split_et_cache, dataset_pour
from src.models import creer_modele, geler_features


def fixer_graines(seed: int) -> None:
    """Reproductibilité : mêmes tirages aléatoires à chaque exécution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def une_epoque(modele, loader, critere, optimiseur, device, scaler, amp, features_gelees, limite=None):
    """Une passe d'entraînement ; renvoie la perte moyenne."""
    modele.train()
    if features_gelees:
        # Les couches gelées gardent aussi leurs statistiques BatchNorm d'ImageNet
        modele.features.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if limite and i >= limite:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimiseur.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = modele(x).squeeze(1)
            perte = critere(logits.float(), y)
        scaler.scale(perte).backward()
        scaler.step(optimiseur)
        scaler.update()
        total += perte.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predire(modele, loader, device, amp=False, critere=None, limite=None):
    """Renvoie (probabilités, labels, perte moyenne ou None)."""
    modele.eval()
    probas, labels, total = [], [], 0.0
    for i, (x, y) in enumerate(loader):
        if limite and i >= limite:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = modele(x).squeeze(1).float()
        if critere is not None:
            total += critere(logits, y).item() * len(y)
        probas.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())
    probas, labels = torch.cat(probas).numpy(), torch.cat(labels).numpy()
    perte = total / len(labels) if critere is not None else None
    return probas, labels, perte


def resume_validation(probas, labels):
    """AUC, sensibilité et spécificité au seuil 0,5 sur la validation."""
    pred = probas >= 0.5
    pos, neg = labels == 1, labels == 0
    return {
        "val_auc": float(roc_auc_score(labels, probas)),
        "val_sensibilite": float(pred[pos].mean()),
        "val_specificite": float((~pred[neg]).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", choices=["baseline", "densenet121"], required=True)
    ap.add_argument("--epochs-tete", type=int, default=0, help="phase 1 : tête seule (DenseNet)")
    ap.add_argument("--lr-tete", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10, help="phase principale (réseau entier)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=3, help="early stopping sur l'AUC de validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="précision mixte float16 (économise la mémoire GPU)")
    ap.add_argument("--cpu", action="store_true", help="forcer le CPU (ex. baseline pendant que le GPU est occupé)")
    ap.add_argument("--limite-batches", type=int, default=None, help="test rapide uniquement")
    args = ap.parse_args()

    fixer_graines(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cpu":
        args.amp = False  # la précision mixte float16 ne concerne que le GPU
    print(f"Appareil : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # ----- Données -----
    df, cache = charger_split_et_cache()
    ds_train = dataset_pour("train", df, cache)
    ds_val = dataset_pour("val", df, cache)
    g = torch.Generator().manual_seed(args.seed)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, generator=g,
                          pin_memory=device.type == "cuda")
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False)

    # ----- Perte pondérée : compense le déséquilibre (~74 % de pneumonies) -----
    # pos_weight = n_normales / n_pneumonies (< 1) : une pneumonie pèse moins dans la
    # perte, une radio normale pèse plus. Les deux classes contribuent autant au total.
    n_pos = int(ds_train.labels.sum())
    n_neg = len(ds_train) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, device=device)
    critere = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Train : {len(ds_train)} images ({n_pos} pneumonies, {n_neg} normales) ; "
          f"val : {len(ds_val)} ; pos_weight = {pos_weight.item():.3f}")

    modele = creer_modele(args.modele).to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    # ----- Phases d'entraînement -----
    # Phase 1 (DenseNet uniquement) : tête seule, features gelées, lr plus grand.
    # Phase 2 : tout le réseau, petit lr (on ajuste finement sans détruire ImageNet).
    phases = []
    if args.epochs_tete > 0:
        phases.append(("tete", args.epochs_tete, args.lr_tete, True))
    phases.append(("complet", args.epochs, args.lr, False))

    DOSSIER_MODELES.mkdir(exist_ok=True)
    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    chemin_modele = DOSSIER_MODELES / f"{args.modele}_meilleur.pt"
    historique, meilleure_auc, meilleure_epoque, epoque = [], -1.0, 0, 0
    debut = time.time()

    for nom_phase, n_epochs, lr, geler in phases:
        if args.modele == "densenet121":
            geler_features(modele, geler)
        params = [p for p in modele.parameters() if p.requires_grad]
        optimiseur = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        planif = torch.optim.lr_scheduler.CosineAnnealingLR(optimiseur, T_max=n_epochs)
        sans_progres = 0
        print(f"\n=== Phase « {nom_phase} » : {n_epochs} époques max, lr={lr:g}, "
              f"{sum(p.numel() for p in params):,} paramètres entraînables ===")

        for _ in range(n_epochs):
            epoque += 1
            t0 = time.time()
            lr_courant = optimiseur.param_groups[0]["lr"]
            perte_train = une_epoque(modele, dl_train, critere, optimiseur, device, scaler,
                                     args.amp, geler, args.limite_batches)
            probas, labels, perte_val = predire(modele, dl_val, device, args.amp, critere)
            planif.step()
            ligne = {"epoque": epoque, "phase": nom_phase, "lr": lr_courant,
                     "perte_train": perte_train, "perte_val": perte_val,
                     **resume_validation(probas, labels), "duree_s": time.time() - t0}
            historique.append(ligne)
            ameliore = ligne["val_auc"] > meilleure_auc
            if ameliore:
                meilleure_auc, meilleure_epoque, sans_progres = ligne["val_auc"], epoque, 0
                torch.save(modele.state_dict(), chemin_modele)
            else:
                sans_progres += 1
            print(f"[{epoque:2d}] {nom_phase:7s} perte train {perte_train:.4f} | val {perte_val:.4f} | "
                  f"AUC {ligne['val_auc']:.4f} | sens {ligne['val_sensibilite']:.3f} | "
                  f"spec {ligne['val_specificite']:.3f} | {ligne['duree_s']:.0f}s"
                  f"{'  <- meilleur' if ameliore else ''}", flush=True)
            with open(DOSSIER_RESULTATS / f"historique_{args.modele}.json", "w", encoding="utf-8") as f:
                json.dump(historique, f, indent=2)
            # L'early stopping ne s'applique qu'à la phase principale
            if not geler and sans_progres >= args.patience:
                print(f"Early stopping : pas d'amélioration de l'AUC depuis {args.patience} époques.")
                break

    resume = {
        "modele": args.modele, "config": vars(args),
        "meilleure_epoque": meilleure_epoque, "meilleure_val_auc": meilleure_auc,
        "epoques_effectuees": epoque, "duree_totale_min": (time.time() - debut) / 60,
        "n_train": len(ds_train), "n_val": len(ds_val), "pos_weight": pos_weight.item(),
        "n_parametres": sum(p.numel() for p in modele.parameters()),
        "appareil": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
    }
    with open(DOSSIER_RESULTATS / f"entrainement_{args.modele}.json", "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2)
    print(f"\nMeilleure AUC de validation : {meilleure_auc:.4f} (époque {meilleure_epoque}) ; "
          f"durée {resume['duree_totale_min']:.1f} min ; modèle : {chemin_modele}")


if __name__ == "__main__":
    main()
