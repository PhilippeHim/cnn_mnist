# Reconnaissance de chiffres manuscrits avec un CNN

Projet d'école autour de la vision par ordinateur et des réseaux de neurones convolutifs avec PyTorch.

L'objectif est d'entraîner un modèle capable de reconnaître des chiffres manuscrits à partir du dataset MNIST, puis de tester ce modèle sur quelques images personnelles.

## Contexte

MNIST est un dataset classique de chiffres manuscrits. Il contient 60 000 images d'entraînement et 10 000 images de test, en niveaux de gris, au format `28 x 28` pixels.

La base MNIST est associée notamment à Yann LeCun, Corinna Cortes et Christopher J.C. Burges. Elle est dérivée d'un ensemble de données plus large produit par le NIST, le National Institute of Standards and Technology. C'est un dataset historique pour apprendre et tester des méthodes de reconnaissance de formes et de machine learning.

Source officielle : https://yann.lecun.org/exdb/mnist/

## Contenu du projet

- `CNN_mnist_.ipynb` : notebook principal du projet.
- `CNN_TP.md` : énoncé ou support de TP associé.
- `cnn_mnist.pth` : poids du modèle entraîné.
- `environment.yml` : environnement Conda recommandé.
- `requirements.txt` : alternative d'installation avec `pip`.

Les données MNIST téléchargées automatiquement dans `data/` ne sont pas versionnées dans Git. Elles seront récupérées au lancement du notebook si nécessaire.

## Installation avec Conda

La méthode recommandée est d'utiliser Conda :

```bash
conda env create -f environment.yml
conda activate computer-vision-mia
python -m ipykernel install --user --name computer-vision-mia --display-name "Python (computer-vision-mia)"
jupyter lab
```

Dans Jupyter, sélectionner ensuite le kernel :

```text
Python (computer-vision-mia)
```

## Installation avec pip

Une alternative est possible avec `requirements.txt` :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Sur Windows, l'activation de l'environnement virtuel se fait plutôt avec :

```bash
.venv\Scripts\activate
```

## Lancer le projet

1. Ouvrir `CNN_mnist_.ipynb`.
2. Sélectionner le bon kernel Python.
3. Exécuter les cellules dans l'ordre.
4. Observer les métriques d'entraînement, de validation et de test.
5. Regarder la matrice de confusion et les images mal classées.
6. Tester les images personnelles si le dossier prévu est disponible.

## Paramètres faciles à modifier

Le notebook a été organisé pour faciliter les essais. Les principaux paramètres sont regroupés dans une cellule dédiée :

```python
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.25
USE_AUGMENTATION = True
AUGMENTATION_DEGREES = 0
MODEL_VERSION = "cnn_3conv"
```

Cela permet de modifier rapidement le comportement du modèle sans chercher dans tout le notebook.

Quelques exemples :

- augmenter `EPOCHS` pour entraîner plus longtemps ;
- modifier `LEARNING_RATE` pour ralentir ou accélérer l'apprentissage ;
- augmenter `DROPOUT_RATE` pour limiter le surapprentissage ;
- activer ou désactiver `USE_AUGMENTATION` ;
- comparer `cnn_2conv` et `cnn_3conv`.

L'idée est de pouvoir "pimper" le modèle progressivement, en observant l'effet de chaque choix sur les résultats.

## Ce qui a été amélioré

Le projet ne se limite pas à entraîner un CNN de base. Plusieurs ajouts ont été faits pour rendre l'expérience plus complète :

- séparation entre train, validation et test ;
- affichage des courbes de loss et d'accuracy ;
- matrice de confusion pour comprendre les erreurs par chiffre ;
- visualisation des prédictions correctes et incorrectes ;
- ajout d'une variante de modèle avec 3 convolutions ;
- ajout du dropout pour limiter le surapprentissage ;
- augmentation légère des images d'entraînement ;
- sauvegarde du modèle entraîné dans `cnn_mnist.pth` ;
- prétraitement d'images personnelles pour essayer le modèle hors MNIST.

Ces modifications apportent une vraie valeur pédagogique : elles permettent de passer d'un simple modèle qui "donne une accuracy" à une démarche plus complète d'expérimentation, de diagnostic et d'amélioration.

## Commentaires pédagogiques

Des commentaires ont été ajoutés directement dans le notebook, à la première personne, pour guider les étudiants qui n'ont pas encore suivi un cours complet sur les CNN.

Ils expliquent notamment :

- le rôle des convolutions ;
- le rôle de `ReLU` ;
- le fonctionnement du max-pooling ;
- le passage de l'image vers un vecteur avec `Flatten` ;
- le rôle des couches linéaires ;
- le dropout ;
- la différence entre `model.train()` et `model.eval()` ;
- la loss, le backward pass et l'optimizer ;
- les dimensions des tenseurs dans le réseau.

Le but est que le notebook puisse être lu comme un support d'apprentissage, pas seulement comme une suite de cellules à exécuter.

## Difficultés rencontrées

Plusieurs difficultés classiques sont apparues pendant le projet :

- comprendre comment les dimensions changent après les convolutions et les max-pooling ;
- éviter de réduire trop vite la taille des images avec trop de pooling ;
- choisir un learning rate stable ;
- limiter le surapprentissage avec le dropout et l'augmentation de données ;
- séparer proprement entraînement, validation et test ;
- interpréter les erreurs du modèle autrement qu'avec une simple accuracy globale ;
- adapter des images personnelles au format MNIST.

La partie images personnelles est particulièrement intéressante : une photo réelle ne ressemble pas directement à une image MNIST. Il faut passer en niveaux de gris, recadrer le chiffre, inverser le fond et le trait, redimensionner en `28 x 28`, puis normaliser comme le dataset d'origine.

## Ce que ce projet nous a apporté

Ce projet a permis de mieux comprendre la logique complète d'un modèle de vision par ordinateur :

- préparer les données ;
- construire une architecture CNN ;
- entraîner le modèle ;
- suivre l'apprentissage avec des métriques ;
- diagnostiquer les erreurs ;
- améliorer progressivement le modèle ;
- sauvegarder et réutiliser un modèle entraîné ;
- tester le modèle sur des données plus réalistes.

Au final, l'intérêt du projet est autant dans le résultat que dans la démarche : on apprend à observer, comparer, modifier, puis mesurer l'effet de chaque amélioration. C'est exactement l'esprit d'un projet de machine learning appliqué.

## Notes Git

Le dépôt ignore les fichiers générés ou trop spécifiques à la machine locale :

- données téléchargées dans `data/` ;
- checkpoints génériques ;
- caches Python et Jupyter ;
- fichiers système comme `.DS_Store`.

Le fichier `cnn_mnist.pth` est volontairement conservé dans Git afin de fournir directement un modèle entraîné avec le projet.
