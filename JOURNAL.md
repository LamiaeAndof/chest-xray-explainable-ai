# Journal du projet

Ce journal liste ce qui a été fait, les problèmes rencontrés et les décisions prises, avec leurs raisons.

## 20 novembre 2026 (soir) : installation

- **Disque C: presque plein.** J'ai installé tout le projet sur D: : environnement virtuel, cache pip, fichiers
  temporaires, poids pré-entraînés (`TORCH_HOME`), caches de matplotlib et de Jupyter.
- **GPU** : NVIDIA MX330 (2 Go) avec le driver 511.69, donc CUDA 11.6 au maximum. Les versions de PyTorch pour
  CUDA 12 exigent un driver plus récent. J'ai donc pris **PyTorch 2.5.1 compilé pour CUDA 11.8**, compatible.
- **Problème** : l'outil `kaggle` plantait (dépendance `six` manquante). J'ai installé `six`.
- **Problème** : la connexion était lente (environ 1 Mo/s), avec 2,3 Go de données et 2,7 Go pour PyTorch. J'ai
  pré-téléchargé les autres paquets en parallèle pour gagner du temps.
- `kaggle.json` est exclu de git dès le début.

## 20 novembre 2026 : exploration (notebook 01)

- 5 856 images, conformes à ce qu'annoncent les auteurs. L'archive Kaggle contient un doublon complet
  (`chest_xray/chest_xray/`) : je n'ai extrait que `train/`, `val/` et `test/`.
- **Problème** : 144 noms de fichiers ne suivaient pas le motif attendu (`IM-0429-0001-0001.jpeg`,
  `person447_virus_921_1.jpeg`). J'ai élargi les motifs ; il reste 0 nom non reconnu.
- **Décision : regroupement prudent des patients.** Les identifiants ne sont pas documentés. 211 numéros existent
  avec et sans le préfixe `NORMAL2-`, et 808 identifiants `personX` du train ont à la fois des images bactériennes
  et virales. J'ai regroupé tout ce qui *pourrait* être le même enfant. *Raison* : regrouper à tort ne crée jamais
  de fuite, alors que séparer à tort peut en créer.
- **Découpage par patient** (`StratifiedGroupKFold`, 1 pli sur 7) : 0 patient commun entre train et validation.
  Pour comparaison, un découpage naïf aurait mis 428 patients des deux côtés.
- **Constat** : 264 identifiants sont communs au train et au test officiels, mais pour les 170 identifiants de
  pneumonie concernés, le sous-type diffère toujours (0 % d'accord contre 43 % attendus au hasard). La
  numérotation semble indépendante entre les splits. J'ai gardé le test officiel intact et je le signale comme
  limite.
- **Découverte : la forme des images trahit la classe.** Le seul rapport largeur/hauteur donne une AUC de 0,865
  sur le train (0,704 sur le test).
- **Décision : recadrage central au lieu d'un redimensionnement en carré.** *Raison* : écraser les images aurait
  transformé cette forme en déformation de l'anatomie, un raccourci facile pour le modèle. Contrepartie
  documentée : les bords latéraux des images très larges sont coupés.
- Pas de flip horizontal (le cœur est à gauche) ni vertical.

## 20-21 novembre 2026 : entraînement (notebook 02)

- **Problème** : DenseNet-121 ne tenait pas en mémoire sur 2 Go, même avec 16 images par batch.
  *Solution* : précision mixte (AMP) et *checkpointing* (`memory_efficient=True`). Le pic mesuré est de 0,60 Go,
  pour environ 23 images/s.
- **Problème** : le DenseNet de torchvision applique un ReLU « in-place » après `norm5`, ce qui peut fausser les
  gradients lus par Grad-CAM. J'ai réécrit le `forward` à l'identique, avec un ReLU normal.
- Pour gagner du temps, la baseline (petit CNN) a été entraînée sur CPU pendant que le GPU entraînait le
  DenseNet.
- Résultats : AUC de validation de 0,9989 pour le DenseNet (époque 9 sur 10, 43,6 min) et de 0,9911 pour la
  baseline (époque 14 sur 15, 38,6 min).

## 21 novembre 2026 : évaluation (notebook 03)

- DenseNet-121 sur le test : AUC 0,976 [0,961–0,988], sensibilité 0,992, spécificité 0,812.
- **Constat honnête** : la spécificité chute de 0,995 en validation à 0,812 en test.
- **Décision** : le seuil « haute sensibilité » (98 % sur la validation) a été fixé *avant* de regarder le test.
  Sur le test, il n'apporte rien au DenseNet. Je le rapporte tel quel.

## 21 novembre 2026 : explicabilité (notebook 04)

- **Problème** : Grad-CAM manquait de mémoire GPU (calcul en float32). J'ai réduit le batch à 2.
- **Problème** : les figures ne s'affichaient pas dans les notebooks (mode `Agg` forcé). Ce mode ne s'applique
  plus qu'aux scripts.
- Analyse : la carte moyenne est centrée sur les poumons pour les pneumonies, et seules 2 cartes sur 624 regardent
  surtout les bords. Une électrode se trouve dans la zone chaude d'un vrai positif.

## 21 novembre 2026 : livrables

- README, poster A4 (1 page, HTML → PDF avec Chrome) et portfolio (GitHub Pages).
- Le portfolio utilise uniquement les données réelles du test (exportées en JSON) pour le curseur de seuil et la
  courbe ROC.
- **Non fait** : la démo dans le navigateur (ONNX Runtime Web). Je n'ai pas pu la tester proprement dans le temps
  imparti. Un script de démo Gradio locale existe (`app/demo_gradio.py`), mais je ne l'ai pas testé.
