# Reconnaissance de chiffres manuscrits avec un CNN

Projet d'école autour de la vision par ordinateur, des réseaux de neurones convolutifs et de PyTorch.

L'objectif est d'entraîner un modèle capable de reconnaître des chiffres manuscrits à partir du dataset MNIST, puis d'analyser finement ses résultats au-delà de la simple accuracy globale.

## Contexte

MNIST est un dataset historique de chiffres manuscrits. Il contient 60 000 images d'entraînement et 10 000 images de test, en niveaux de gris, au format `28 x 28` pixels.

La base MNIST est associée notamment à Yann LeCun, Corinna Cortes et Christopher J.C. Burges. Elle est dérivée d'un ensemble de données plus large produit par le NIST, le National Institute of Standards and Technology. Elle reste aujourd'hui un dataset de référence pour découvrir la reconnaissance de formes, le machine learning et les premiers modèles de vision par ordinateur.

Source officielle : https://yann.lecun.org/exdb/mnist/

## Objectifs du projet

Ce projet avait plusieurs objectifs :

- construire un CNN capable de reconnaître les chiffres de `0` à `9` ;
- comprendre le rôle des couches de convolution, du max-pooling, du dropout et des couches linéaires ;
- suivre l'entraînement avec des métriques lisibles ;
- comparer les performances globales et les performances par classe ;
- améliorer progressivement le modèle en ajustant ses paramètres ;
- documenter le notebook pour le rendre accessible à des étudiants qui découvrent les CNN.

## Contenu du dépôt

- `CNN_mnist_.ipynb` : notebook principal du projet.
- `CNN_TP.md` : support ou énoncé de TP associé.
- `cnn_mnist.pth` : poids du modèle entraîné, volontairement conservés dans Git.
- `environment.yml` : environnement Conda recommandé.
- `requirements.txt` : alternative d'installation avec `pip`.

Les données MNIST téléchargées automatiquement dans `data/` ne sont pas versionnées. Elles seront récupérées au lancement du notebook si nécessaire.

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

Une installation avec `pip` est aussi possible :

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

## Lancer le notebook

1. Ouvrir `CNN_mnist_.ipynb`.
2. Sélectionner le bon kernel Python.
3. Exécuter les cellules dans l'ordre.
4. Observer les métriques d'entraînement, de validation et de test.
5. Lire la matrice de confusion.
6. Comparer les scores par classe.
7. Observer les images mal classées pour comprendre les erreurs du modèle.

## Paramètres faciles à modifier

Le notebook a été organisé pour faciliter les expérimentations. Les principaux paramètres sont regroupés dans une cellule dédiée :

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

L'idée est de pouvoir améliorer le modèle progressivement, en observant l'effet réel de chaque choix sur les résultats.

## Commentaires pédagogiques

Des commentaires ont été ajoutés directement dans le notebook pour guider les étudiants qui n'ont pas encore suivi un cours complet sur les CNN.

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

## Fonctionnement du notebook CNN sur MNIST

Le notebook `CNN_mnist_.ipynb` constitue le cœur du projet. Il met en place un pipeline complet de deep learning : préparation des données, construction du modèle, entraînement, évaluation, diagnostic des erreurs et sauvegarde du modèle.

L'idée n'est pas seulement d'obtenir un bon score, mais de comprendre ce qui se passe à chaque étape.

### Vérification des dépendances

Le notebook commence par vérifier que les bibliothèques nécessaires sont bien installées :

```python
required_packages = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "torchvision": "torchvision",
}
```

Cette cellule rend le notebook plus simple à exécuter avec `Run All`. Si une dépendance manque, elle est installée dans l'environnement Python actif du notebook.

Une vérification spécifique est aussi faite pour NumPy :

```python
if numpy_major >= 2:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2"])
```

Cette précaution vient du fait que certaines versions de PyTorch sont sensibles aux versions de NumPy. Le but est d'éviter une erreur d'import dès le début du TP.

### Choix du device

Le notebook choisit automatiquement le meilleur matériel disponible :

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

Cela permet d'utiliser :

- `mps` sur les Mac Apple Silicon ;
- `cuda` sur une carte graphique NVIDIA ;
- `cpu` si aucune accélération GPU n'est disponible.

Ce choix est important, car l'entraînement d'un réseau de neurones peut être beaucoup plus rapide sur GPU.

### Reproductibilité

Deux graines aléatoires sont fixées :

```python
torch.manual_seed(42)
np.random.seed(42)
```

Cela permet de rendre les résultats plus stables d'une exécution à l'autre. Sans cela, le découpage train/validation, l'ordre des images ou l'initialisation des poids peuvent changer légèrement à chaque lancement.

### Centralisation des paramètres

Les principaux paramètres sont regroupés dans une seule cellule :

```python
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.25
USE_AUGMENTATION = True
AUGMENTATION_DEGREES = 0
MODEL_VERSION = "cnn_3conv"
```

Cette organisation facilite l'expérimentation. Au lieu de modifier le code à plusieurs endroits, on peut ajuster les paramètres principaux au même endroit.

Chaque paramètre a un rôle précis :

- `EPOCHS` : nombre de passages complets sur le jeu d'entraînement ;
- `BATCH_SIZE` : nombre d'images traitées avant une mise à jour des poids ;
- `LEARNING_RATE` : taille du pas d'apprentissage ;
- `DROPOUT_RATE` : proportion de neurones désactivés pendant l'entraînement ;
- `USE_AUGMENTATION` : activation ou non de l'augmentation de données ;
- `AUGMENTATION_DEGREES` : angle maximal de rotation pendant l'augmentation ;
- `MODEL_VERSION` : choix entre le CNN à 2 convolutions et celui à 3 convolutions.

### Transformation des images MNIST

Les images MNIST sont chargées avec `torchvision.datasets.MNIST`. Avant d'être données au modèle, elles sont transformées :

```python
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

La transformation `ToTensor()` convertit l'image en tenseur PyTorch. Une image MNIST devient un tenseur de forme :

```text
1 x 28 x 28
```

Le `1` correspond au canal unique de l'image, car MNIST est en niveaux de gris.

La normalisation :

```python
transforms.Normalize((0.1307,), (0.3081,))
```

centre et réduit les valeurs avec la moyenne et l'écart-type classiques de MNIST. Cela aide le modèle à apprendre plus proprement, car les valeurs d'entrée sont placées sur une échelle plus stable.

### Augmentation de données

Le notebook prévoit aussi une transformation augmentée :

```python
augmented_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=AUGMENTATION_DEGREES,
        translate=(0.08, 0.08),
        scale=(0.98, 1.10)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

L'augmentation de données consiste à modifier légèrement les images d'entraînement pour rendre le modèle plus robuste.

Ici, plusieurs variations peuvent être appliquées :

- petite rotation ;
- légère translation horizontale ou verticale ;
- léger changement d'échelle.

L'idée est simple : un chiffre reste le même même s'il est un peu décalé, un peu plus grand, un peu plus petit ou légèrement incliné. En voyant ces variations pendant l'entraînement, le modèle apprend à mieux généraliser.

L'augmentation est réservée au jeu d'entraînement :

```python
train_transform = augmented_transform if USE_AUGMENTATION else basic_transform
```

La validation et le test restent sans augmentation. C'est important, car on veut mesurer le modèle sur des données stables, pas sur des images modifiées aléatoirement à chaque exécution.

### Découpage train, validation et test

Le dataset MNIST contient déjà un train set et un test set. Le notebook ajoute un jeu de validation en prélevant 10 % du train set :

```python
VALIDATION_RATIO = 0.1
validation_size = int(len(mnist_train_full_basic) * VALIDATION_RATIO)
train_size = len(mnist_train_full_basic) - validation_size
```

On obtient donc trois ensembles :

- `train` : sert à apprendre les poids du modèle ;
- `validation` : sert à suivre l'apprentissage pendant les essais ;
- `test` : sert à mesurer la performance finale.

Cette séparation évite de juger le modèle uniquement sur les images qu'il a utilisées pour apprendre.

### DataLoaders et batches

Les `DataLoader` découpent les données en lots d'images :

```python
train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(mnist_validation, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)
```

Un batch est un paquet d'images envoyé au modèle en une seule fois. Avec `BATCH_SIZE = 64`, le modèle traite 64 images avant de mettre à jour ses poids.

Le train loader utilise `shuffle=True` pour mélanger les images à chaque epoch. Cela évite que le modèle voie toujours les chiffres dans le même ordre.

### Vérification de la répartition des classes

Le notebook compte le nombre d'images pour chaque chiffre :

```python
def count_classes(dataset, num_classes=10):
    counts = np.zeros(num_classes, dtype=int)
    for _, label in dataset:
        counts[label] += 1
    return counts
```

Cette étape permet de vérifier que les classes sont correctement représentées. Si un chiffre était beaucoup moins présent que les autres, le modèle pourrait moins bien l'apprendre.

### Visualisation des images

Avant d'entraîner le modèle, le notebook affiche quelques images MNIST :

```python
plt.imshow(image, cmap="gray")
```

Cette étape paraît simple, mais elle est importante : elle permet de vérifier ce que le modèle va réellement recevoir. En machine learning, il faut toujours regarder les données avant de faire confiance aux résultats.

Comme les images ont été normalisées, une fonction permet de les remettre dans une échelle affichable :

```python
def denormalize_mnist(image):
    return image * 0.3081 + 0.1307
```

### Architecture du CNN

Deux architectures sont disponibles dans le notebook :

- `MNISTCNN2Conv` : version simple avec 2 convolutions ;
- `MNISTCNN3Conv` : version plus expressive avec 3 convolutions.

Un CNN est divisé en deux grandes parties :

- `features` : extraction des caractéristiques visuelles ;
- `classifier` : transformation de ces caractéristiques en prédiction finale.

Dans la version à 3 convolutions, le trajet de l'image est le suivant :

```text
Entrée : 1 x 28 x 28
-> Conv 1 : 16 x 28 x 28
-> ReLU
-> MaxPool : 16 x 14 x 14
-> Conv 2 : 32 x 14 x 14
-> ReLU
-> MaxPool : 32 x 7 x 7
-> Conv 3 : 64 x 7 x 7
-> ReLU
-> Flatten : 64 * 7 * 7
-> Linear : 128
-> ReLU
-> Dropout
-> Linear : 10
```

La dernière couche produit 10 scores, un pour chaque chiffre de `0` à `9`.

### Rôle des convolutions

Une convolution apprend des filtres capables de détecter de petits motifs dans l'image :

- traits ;
- bords ;
- courbes ;
- coins ;
- formes locales.

Dans le notebook, les convolutions utilisent :

```python
nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
```

Cela signifie :

- `in_channels=1` : l'image d'entrée a un seul canal ;
- `out_channels=16` : le modèle apprend 16 filtres différents ;
- `kernel_size=3` : chaque filtre regarde une zone de `3 x 3` pixels ;
- `padding=1` : on conserve la même taille spatiale après la convolution.

Au fil des couches, le modèle passe de motifs simples à des motifs plus abstraits.

### Rôle de ReLU

Après chaque convolution, le notebook utilise `ReLU` :

```python
nn.ReLU()
```

ReLU introduit de la non-linéarité. Sans cette étape, le réseau serait beaucoup moins capable d'apprendre des relations complexes.

De manière simple, ReLU garde les valeurs positives et remplace les valeurs négatives par zéro.

### Rôle du max-pooling

Le max-pooling réduit la taille de l'image :

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

Avec ce paramétrage :

```text
28 x 28 -> 14 x 14
14 x 14 -> 7 x 7
```

Le max-pooling garde l'information la plus forte dans chaque petite zone. Cela réduit le nombre de calculs et rend le modèle un peu plus robuste aux petits décalages.

### Rôle de Flatten et des couches linéaires

Après les convolutions, les données ont encore une forme spatiale :

```text
64 x 7 x 7
```

La couche `Flatten` transforme cette sortie en vecteur :

```python
nn.Flatten()
```

Puis les couches linéaires prennent la décision finale :

```python
nn.Linear(64 * 7 * 7, 128)
nn.Linear(128, 10)
```

La première couche résume les caractéristiques dans 128 neurones. La dernière couche produit les 10 scores de classification.

### Rôle du dropout

Le dropout est utilisé juste avant la dernière couche :

```python
nn.Dropout(p=dropout_rate)
```

Pendant l'entraînement, il désactive aléatoirement une partie des neurones. Avec `DROPOUT_RATE = 0.25`, environ 25 % des activations sont mises à zéro à chaque passage.

Le but est de limiter le surapprentissage. Le modèle ne peut pas dépendre toujours des mêmes neurones : il doit apprendre des représentations plus générales.

En mode évaluation, le dropout est désactivé automatiquement avec :

```python
model.eval()
```

### Vérification des dimensions internes

Le notebook crée un faux batch pour vérifier les dimensions :

```python
dummy_batch = torch.randn(4, 1, 28, 28).to(device)
```

Cette vérification permet de confirmer que la sortie des convolutions et la sortie finale ont les formes attendues :

- sortie des convolutions ;
- sortie du modèle ;
- 10 scores par image.

C'est une étape très utile pour éviter les erreurs de dimension, fréquentes quand on construit un CNN.

### Loss et optimizer

La fonction de perte utilisée est :

```python
criterion = nn.CrossEntropyLoss()
```

Elle est adaptée à une classification multi-classes. Le modèle produit 10 scores, et la loss mesure l'écart entre ces scores et la vraie classe.

L'optimiseur utilisé est Adam :

```python
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

Adam ajuste les poids du modèle pour faire diminuer la loss.

### Boucle d'entraînement

La fonction `train_one_epoch` entraîne le modèle pendant une epoch :

```python
model.train()
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

Les étapes sont les suivantes :

1. passer le modèle en mode entraînement ;
2. remettre les gradients à zéro ;
3. faire passer les images dans le modèle ;
4. calculer la loss ;
5. calculer les gradients avec `backward()` ;
6. mettre à jour les poids avec `optimizer.step()`.

Une epoch correspond à un passage complet sur toutes les images d'entraînement.

### Évaluation du modèle

La fonction `evaluate` mesure les performances sans modifier les poids :

```python
model.eval()
with torch.no_grad():
    outputs = model(images)
```

Deux éléments sont importants :

- `model.eval()` désactive certains comportements d'entraînement, comme le dropout ;
- `torch.no_grad()` évite de calculer les gradients, ce qui économise de la mémoire.

Cette fonction est utilisée pour la validation et le test.

### Courbes d'apprentissage

Le notebook conserve l'historique de la loss et de l'accuracy :

```python
history = {
    "train_loss": [],
    "train_accuracy": [],
    "validation_loss": [],
    "validation_accuracy": [],
}
```

Les courbes permettent de comprendre le comportement du modèle :

- si la loss baisse, le modèle apprend ;
- si l'accuracy monte, le modèle prédit mieux ;
- si le train progresse mais pas la validation, il peut y avoir surapprentissage ;
- si tout stagne, les paramètres ou l'architecture peuvent être à revoir.

### Évaluation sur le test set

Une fois l'entraînement terminé, le modèle est évalué sur le test set :

```python
test_loss, test_accuracy, test_predictions, test_labels = evaluate(
    model,
    test_loader,
    criterion,
    device
)
```

Le test set sert à obtenir une mesure finale sur des images que le modèle n'a pas utilisées pour apprendre ni pour ajuster les paramètres.

### Matrice de confusion et précision par classe

Le notebook construit une matrice de confusion :

```python
matrix[true_label, pred_label] += 1
```

Cette matrice permet de voir quelles erreurs le modèle fait réellement. Par exemple, elle permet de repérer si des `9` sont souvent prédits comme des `8`, ou si un chiffre est moins bien reconnu que les autres.

Le notebook calcule aussi l'accuracy par classe :

```python
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
```

Cette étape est essentielle dans le projet. Elle permet de ne pas se contenter d'une très bonne accuracy globale et de vérifier si chaque chiffre est correctement reconnu.

### Visualisation des prédictions et des erreurs

Le notebook affiche quelques prédictions :

```python
show_predictions(model, test_loader, device, n_images=12)
```

Il affiche aussi les erreurs :

```python
show_mistakes(model, test_loader, device, max_images=20)
```

Cette étape rend le diagnostic beaucoup plus concret. Une erreur peut être compréhensible si le chiffre est mal écrit, ambigu ou proche d'un autre chiffre.

### Sauvegarde du modèle

Le modèle entraîné est sauvegardé avec :

```python
torch.save(model.state_dict(), "cnn_mnist.pth")
```

Le fichier `cnn_mnist.pth` contient les poids appris par le modèle. Il permet de réutiliser le modèle sans refaire tout l'entraînement.

### Test sur des images personnelles

Le notebook contient aussi une section pour tester des images personnelles rangées dans `df/test`.

Comme une photo réelle ne ressemble pas directement à MNIST, un prétraitement spécifique est appliqué :

- passage en niveaux de gris ;
- correction éventuelle de l'orientation EXIF ;
- amélioration du contraste ;
- suppression d'une zone haute parasite ;
- détection des pixels foncés correspondant à l'encre ;
- recadrage autour du chiffre ;
- conversion au style MNIST : fond noir, chiffre blanc ;
- centrage dans une image carrée ;
- redimensionnement en `28 x 28` ;
- normalisation avec les statistiques MNIST.

Cette étape montre une difficulté importante : un modèle entraîné sur MNIST fonctionne très bien sur MNIST, mais les images réelles nécessitent souvent une adaptation pour ressembler au format d'entraînement.

### Diagnostic de l'orientation EXIF

Certaines photos prises avec un smartphone peuvent apparaître tournées selon le logiciel utilisé. Le notebook ajoute donc une cellule de diagnostic :

```python
debug_image_orientation("df/test/0/0_008.jpeg")
```

Elle affiche :

- l'image brute ;
- l'image avec orientation EXIF appliquée ;
- l'image prétraitée au format MNIST.

Cela permet de comprendre si une erreur vient du modèle ou simplement d'une image mal orientée avant le prétraitement.

## Démarche d'amélioration du modèle

Lors du premier lancement, le modèle a rapidement obtenu un très bon score global, proche de 98 %. Cependant, l'accuracy globale ne suffisait pas à juger finement la qualité du modèle : certaines classes étaient moins bien reconnues que d'autres.

Le notebook affiche donc aussi la précision par classe, c'est-à-dire le score obtenu pour chaque chiffre de `0` à `9`. Il indique également le chiffre le mieux détecté et le chiffre le moins bien détecté. Cette analyse permet de ne pas seulement regarder la moyenne générale, mais aussi l'équilibre entre les différentes classes.

Même avec une accuracy globale autour de 99,4 %, le chiffre le moins bien détecté restait le `9`, avec environ 98,2 % de précision. L'objectif est alors devenu plus précis : obtenir un modèle performant en moyenne, mais aussi plus régulier sur l'ensemble des chiffres.

### Ajustement des epochs

Le nombre d'epochs a été testé progressivement : 5 au départ, puis 8, 10 et enfin 12.

En observant les courbes d'accuracy et de loss, l'accuracy montait très vite dès la première epoch, autour de 97 %, puis ralentissait progressivement avant de se stabiliser vers la 8e epoch. Continuer jusqu'à 12 epochs a tout de même permis d'améliorer les scores par classe et de mieux équilibrer les résultats.

### Ajustement des hyperparamètres

Plusieurs paramètres ont ensuite été ajustés :

- le `learning_rate`, pour contrôler la vitesse d'apprentissage ;
- le `dropout_rate`, pour limiter le surapprentissage ;
- l'augmentation de données, notamment l'angle de rotation des images.

Le dropout joue ici un rôle important : avec un taux de 25 %, il désactive aléatoirement une partie des neurones pendant l'entraînement. Cela oblige le modèle à apprendre de manière plus générale, au lieu de trop dépendre de certains détails spécifiques du jeu d'entraînement.

Plusieurs angles de rotation ont aussi été testés pour l'augmentation de données : 10 degrés au départ, puis 12, puis 9. Cette étape a permis d'améliorer la détection du chiffre `9`, qui était la classe la moins bien reconnue au départ.

### Ajout d'une troisième convolution

L'amélioration la plus importante est venue de l'ajout d'une troisième convolution. L'architecture est alors devenue :

```text
Entrée : 1 x 28 x 28
-> Conv 1 : 16 x 28 x 28
-> MaxPool : 16 x 14 x 14
-> Conv 2 : 32 x 14 x 14
-> MaxPool : 32 x 7 x 7
-> Conv 3 : 64 x 7 x 7
-> Flatten : 64 * 7 * 7
-> Linear : 128
-> Linear : 10
```

Cette convolution supplémentaire permet au modèle d'apprendre des caractéristiques plus riches avant la classification finale. Elle a fait progresser la détection du chiffre `9` jusqu'à environ 99 %. Le chiffre le moins bien détecté est alors devenu le `8`, ce qui montre que le modèle s'est mieux équilibré sur l'ensemble des classes.

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
- analyse détaillée des scores par classe.

Ces modifications apportent une vraie valeur pédagogique : elles permettent de passer d'un simple modèle qui donne une accuracy à une démarche plus complète d'expérimentation, de diagnostic et d'amélioration.

## Difficultés rencontrées

Plusieurs difficultés classiques sont apparues pendant le projet :

- comprendre comment les dimensions changent après les convolutions et les max-pooling ;
- éviter de réduire trop vite la taille des images avec trop de pooling ;
- choisir un learning rate stable ;
- limiter le surapprentissage avec le dropout et l'augmentation de données ;
- séparer proprement entraînement, validation et test ;
- interpréter les erreurs du modèle autrement qu'avec une simple accuracy globale ;
- équilibrer les performances entre les différentes classes.

Le projet a aussi montré qu'une très bonne accuracy globale peut masquer des différences entre les classes. C'est pour cela que l'analyse par chiffre est importante : elle permet de repérer les classes qui restent plus fragiles et de guider les améliorations.

## Préparation des données et détection des contours

Le projet inclut aussi un travail préparatoire autour de la simplification de l'image, présenté dans `contours_seuillage_gradient.ipynb`. L'idée générale est de transformer une image riche, composée de milliers de nuances et de détails, en une représentation plus simple : zones claires, zones sombres, contours et formes.

Cette étape est importante en vision par ordinateur, car un ordinateur ne "voit" pas une image comme nous. Il manipule une matrice de pixels. Chaque pixel possède une valeur, par exemple entre `0` et `255` pour une image en niveaux de gris :

- `0` correspond au noir ;
- `255` correspond au blanc ;
- les valeurs intermédiaires correspondent aux nuances de gris.

Le notebook explore plusieurs techniques classiques pour passer d'une image brute à une image plus facile à analyser.

### Passage en niveaux de gris

La première simplification consiste à convertir l'image couleur en niveaux de gris :

```python
img = cv2.imread("images/normandie.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

Cela permet de supprimer l'information de couleur pour ne garder que l'intensité lumineuse. On passe donc d'une image avec trois canaux couleur à une seule matrice de valeurs de gris. C'est souvent plus simple pour détecter des formes, des contrastes ou des contours.

### Analyse de l'histogramme

Le notebook commence par analyser la répartition des valeurs de gris avec un histogramme :

```python
plt.hist(gray.flatten(), 256, [0, 256])
```

La méthode `flatten()` transforme la matrice de l'image en une longue liste de pixels. L'histogramme indique ensuite combien de pixels possèdent chaque valeur de gris entre `0` et `255`.

Cette étape permet d'identifier les zones sombres et les zones claires de l'image. Si l'histogramme présente deux grands groupes de valeurs, on peut chercher une "vallée" entre les deux : cette vallée correspond à un seuil possible pour séparer le clair du sombre.

### Seuillage manuel

Le seuillage consiste à choisir une valeur limite. Les pixels d'un côté du seuil deviennent blancs, les autres deviennent noirs. On obtient alors une image binaire, composée uniquement de `0` et de `255`.

Dans le notebook, le seuil est d'abord trouvé manuellement à partir de l'histogramme :

```python
counts, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
pic_1 = 65
pic_2 = 140
vallee = counts[pic_1:pic_2]
index_du_minimum = np.argmin(vallee)
seuil = index_du_minimum + pic_1
```

L'idée est de regarder entre deux pics de l'histogramme et de choisir la valeur où il y a le moins de pixels. Cette valeur marque une séparation naturelle entre deux groupes de luminosité.

Ensuite, l'image est binarisée :

```python
image_binarisée = np.where(gray < seuil, 255, 0).astype("uint8")
```

Ici, les pixels plus sombres que le seuil deviennent blancs (`255`) et les autres deviennent noirs (`0`). Cette inversion est utile quand on veut faire ressortir les éléments sombres de l'image comme des formes blanches sur fond noir.

### Seuillage automatique avec Otsu

Le notebook montre aussi une méthode automatique : le seuillage d'Otsu.

```python
seuil_calcule, thresh_otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

Otsu cherche automatiquement le meilleur seuil pour séparer deux groupes de pixels. Il évite de choisir le seuil à la main. Dans le notebook, le seuil trouvé manuellement est proche de celui calculé par Otsu, ce qui confirme que l'analyse de l'histogramme était cohérente.

Cette technique est très utilisée lorsque l'image présente un contraste assez net entre le fond et les objets.

### Détection des contours avec `findContours`

Une fois l'image binarisée, OpenCV peut chercher les contours :

```python
contours, _ = cv2.findContours(
    image_binarisée,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)
```

La fonction `findContours` parcourt l'image binaire et repère les frontières des zones blanches. Elle suit les pixels connectés pour reconstruire les contours des formes.

Deux paramètres sont importants :

- `cv2.RETR_LIST` récupère tous les contours sans construire de hiérarchie ;
- `cv2.CHAIN_APPROX_SIMPLE` simplifie les contours en supprimant les points redondants.

Sans cette simplification, un contour pourrait contenir énormément de points, parfois inutiles. Avec `CHAIN_APPROX_SIMPLE`, OpenCV garde seulement les points nécessaires pour décrire la forme de manière plus compacte.

Les contours sont ensuite affichés sur l'image originale :

```python
cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
```

Le `-1` signifie que l'on dessine tous les contours détectés.

### Réduction du bruit avec le flou gaussien

Le notebook montre ensuite qu'il peut être utile de "dégrader" légèrement l'image avant de détecter les contours :

```python
gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
```

Cela peut sembler paradoxal, mais le flou gaussien améliore souvent la détection. Il lisse les petits détails et réduit le bruit. Résultat : les contours détectés sont parfois plus propres, car OpenCV se concentre davantage sur les grandes formes plutôt que sur les micro-variations de pixels.

Le noyau `(7, 7)` indique la taille de la zone utilisée pour lisser chaque pixel. Plus le noyau est grand, plus l'image est floutée.

### Création de masques

Le notebook crée aussi des masques noirs pour afficher uniquement les contours ou les formes remplies :

```python
mask_traits = np.zeros(gray.shape, dtype="uint8")
mask_rempli = np.zeros(gray.shape, dtype="uint8")

cv2.drawContours(mask_traits, contours, -1, 255, 2)
cv2.drawContours(mask_rempli, contours, -1, 255, -1)
```

Le premier masque affiche seulement les traits des contours. Le second remplit complètement les formes détectées.

C'est une étape intéressante, car elle montre la différence entre :

- détecter une frontière ;
- reconstruire une zone complète ;
- isoler une forme du reste de l'image.

Cette logique de masque est très utilisée en segmentation d'image.

### Détection par gradient avec Sobel

La deuxième méthode étudiée dans le notebook repose sur le gradient. Contrairement au seuillage, qui cherche à séparer des zones claires et sombres, le gradient cherche les ruptures brutales d'intensité.

Autrement dit, il répond à la question : "où est-ce que l'image change très vite ?"

Le notebook utilise Sobel :

```python
grad_x = cv2.Sobel(gray_blurred, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray_blurred, cv2.CV_32F, 0, 1, ksize=3)
```

- `grad_x` détecte les variations horizontales, donc les bords plutôt verticaux ;
- `grad_y` détecte les variations verticales, donc les bords plutôt horizontaux.

On combine ensuite ces deux gradients pour calculer une magnitude :

```python
mag = np.sqrt(grad_x**2 + grad_y**2)
```

Cette magnitude indique la force du contour. Plus la valeur est élevée, plus la rupture de luminosité est forte.

La magnitude est ensuite normalisée entre `0` et `255` :

```python
mag_norm = np.uint8(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))
```

Puis un seuil est appliqué :

```python
_, thresh_binary = cv2.threshold(mag_norm, 46, 255, cv2.THRESH_BINARY)
```

On obtient une nouvelle image binaire, mais cette fois basée sur les changements brusques de luminosité plutôt que sur les zones claires ou sombres.

### Détection de contours avec Canny

Enfin, le notebook utilise Canny, une méthode plus avancée de détection de contours :

```python
edges_canny = cv2.Canny(gray_blurred, seuil_bas, seuil_haut)
```

Canny repose notamment sur Sobel, mais ajoute une analyse plus fine. Sa grande différence est l'utilisation de deux seuils :

- sous le seuil bas, le gradient est considéré comme trop faible : pas de contour ;
- au-dessus du seuil haut, le gradient est considéré comme fort : contour certain ;
- entre les deux, le pixel est gardé seulement s'il est connecté à un contour fort.

Dans le notebook, les seuils sont calculés automatiquement à partir de la médiane de l'image :

```python
v = np.median(gray_blurred)
sigma = 0.33

seuil_bas = int(max(0, (1.0 - sigma) * v))
seuil_haut = int(min(255, (1.0 + sigma) * v))
```

Cette approche évite de fixer les seuils totalement au hasard. Elle adapte les valeurs à la luminosité générale de l'image.

### Limites observées

Ces techniques sont très utiles pour comprendre les bases de la vision par ordinateur, mais elles ont aussi des limites :

- le seuillage fonctionne bien si le contraste entre l'objet et le fond est clair ;
- il devient moins fiable si l'éclairage varie beaucoup ;
- le flou aide à réduire le bruit, mais peut aussi supprimer certains détails ;
- Sobel détecte les ruptures, mais peut produire beaucoup de petits contours ;
- Canny est plus robuste, mais dépend encore du choix des seuils.

Cette partie du projet montre donc pourquoi les approches classiques restent importantes : elles permettent de comprendre ce que signifie "préparer une image" avant de passer à des méthodes plus avancées comme les CNN.

## Ce que ce projet nous a apporté

Ce projet a permis de mieux comprendre la logique complète d'un modèle de vision par ordinateur :

- préparer les données en simplifiant l'image : niveaux de gris, seuillage, contours, masques et gradients ;
- construire une architecture CNN ;
- entraîner le modèle ;
- suivre l'apprentissage avec des métriques ;
- diagnostiquer les erreurs ;
- améliorer progressivement le modèle ;
- sauvegarder et réutiliser un modèle entraîné ;
- comparer la performance globale et la performance par classe.

Au final, l'intérêt du projet est autant dans le résultat que dans la démarche : on apprend à observer, comparer, modifier, puis mesurer l'effet de chaque amélioration. Comme chaque projet de machine learning appliqué.

## Notes Git

Le dépôt ignore les fichiers générés ou trop spécifiques à la machine locale :

- données téléchargées dans `data/` ;
- checkpoints génériques ;
- caches Python et Jupyter ;
- fichiers système comme `.DS_Store`.

Le fichier `cnn_mnist.pth` est volontairement conservé dans Git afin de fournir directement un modèle entraîné avec le projet.
