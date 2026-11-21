"""Modèles : baseline (petit CNN entraîné de zéro) et DenseNet-121 pré-entraîné sur ImageNet.

Les deux modèles renvoient UN seul logit par image : sigmoid(logit) = P(pneumonie).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class PetitCNN(nn.Module):
    """Baseline : 4 blocs [conv 3x3 -> BatchNorm -> ReLU -> MaxPool], puis moyenne globale.

    Environ 100 000 paramètres (contre ~7 millions pour DenseNet-121), sans aucun
    pré-entraînement. Sert de point de comparaison : que gagne-t-on avec un grand
    réseau pré-entraîné ?
    """

    def __init__(self, canaux=(16, 32, 64, 128)):
        super().__init__()
        couches, entree = [], 3
        for sortie in canaux:
            couches += [
                nn.Conv2d(entree, sortie, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(sortie),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            entree = sortie
        self.features = nn.Sequential(*couches)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(entree, 1))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(x)


class DenseNet121Binaire(nn.Module):
    """DenseNet-121 (Huang et al., 2017), le réseau utilisé par CheXNet (Rajpurkar et al., 2017).

    - `features` : l'extracteur de torchvision, pré-entraîné sur ImageNet.
    - `classifier` : la tête ImageNet (1000 classes) est remplacée par 1 sortie.
    Le forward est réécrit à l'identique de torchvision, mais avec un ReLU NON
    « in-place » : le ReLU in-place de torchvision modifie la sortie de norm5 sur
    place, ce qui peut fausser les gradients récupérés par Grad-CAM.
    Les noms des couches (features.*, classifier.*) sont ceux de torchvision.

    memory_efficient=True (« checkpointing ») : les activations intermédiaires des
    blocs denses sont recalculées pendant la rétropropagation au lieu d'être
    stockées. Indispensable sur un GPU de 2 Go ; mêmes résultats, un peu plus lent.
    """

    def __init__(self, pretrained: bool = True, memory_efficient: bool = True):
        super().__init__()
        poids = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.densenet121(weights=poids, memory_efficient=memory_efficient)
        self.features = base.features
        self.classifier = nn.Linear(base.classifier.in_features, 1)

    def forward(self, x):
        x = F.relu(self.features(x))
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return self.classifier(x)


def creer_densenet121(pretrained: bool = True) -> nn.Module:
    return DenseNet121Binaire(pretrained)


def creer_modele(nom: str, pretrained: bool = True) -> nn.Module:
    if nom == "baseline":
        return PetitCNN()
    if nom == "densenet121":
        return creer_densenet121(pretrained)
    raise ValueError(f"Modèle inconnu : {nom}")


def geler_features(modele: nn.Module, geler: bool) -> None:
    """Gèle (ou dégèle) l'extracteur de caractéristiques ; la tête reste entraînable."""
    for p in modele.features.parameters():
        p.requires_grad = not geler


def couche_cible_gradcam(modele: nn.Module) -> nn.Module:
    """Dernière couche « spatiale » avant la moyenne globale, utilisée par Grad-CAM.

    - DenseNet-121 de torchvision : features = [..., denseblock4, norm5] ; le forward
      applique ensuite ReLU puis la moyenne globale. On prend norm5 (= features[-1]) :
      c'est la sortie du dernier bloc dense (7x7x1024), juste avant la moyenne.
    - PetitCNN : on prend le dernier bloc convolutif avant le MaxPool final (ReLU).
    """
    if isinstance(modele, PetitCNN):
        return modele.features[-2]
    return modele.features[-1]


def charger_modele(nom: str, chemin, device="cpu") -> nn.Module:
    """Recharge un modèle entraîné (sans retélécharger les poids ImageNet)."""
    modele = creer_modele(nom, pretrained=False)
    etat = torch.load(chemin, map_location=device, weights_only=True)
    modele.load_state_dict(etat)
    return modele.to(device).eval()
