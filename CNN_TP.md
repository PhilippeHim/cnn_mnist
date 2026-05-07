# TP3 — Construction d'un CNN 
## Données

- Importer vos données avec torchvision de PyTorch : 
mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
- Afficher la shape du dataset 
- Afficher la shape d'une image
- Visualiser la réparition des classes dans votre train set et votre test set 
- Afficher les 50 premières images. 
---

## Model

### 1. Sequence
- Créez votre séquence avec vos couches de convolution, vos fonctions d'activation, vos couches de pooling et vos couche fully connected jusqu'à la couche de sortie 

---

### 2. Loss Function et Optimisateur 
Utilisez la fonction de loss et l'optimizer suivants : 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 

---

### 3. Epochs
- Faites l'apprentissage de votre modèle en choisissant un nombre d'Epochs. 
- Vous afficherez la perte à chaque Epochs. 

---

### 4. Evaluation du modèle
- Evaluez votre modèle sur les données de test 
- Donnez la matrice de confusion
- Quels chiffres sont le mieux détectés par votre modèle, le moins bien ? 

---

### 5. Diagnostic
- Affichez les images pour lesquels votre modèle s'est trompé
- Les erreurs sont elles cohérentes ? 

---

### 6. Test réel
- Importez les images manuscrites de vos camarades 
- Faites le prétraitement necessaire pour que les images soit de la même nature que vos images d'entrainement
- Tester votre modèle sur ces images. 
- Affichez les erreurs, diagnostiquez, interprétez. 

