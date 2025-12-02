# Hetic_KerasMachineLearning

Création d'un classifieur de cancer du sein à l'aide de l'API Keras/TensorFlow.

## Prérequis

- Python 3.12 (TensorFlow n'est pas compatible avec Python 3.13+)

## Installation

1. Créer l'environnement virtuel avec Python 3.12 :

```powershell
py -3.12 -m venv venv
```

2. Installer les dépendances :

```powershell
.\venv\Scripts\Activate.ps1
pip install tensorflow numpy scikit-learn matplotlib
```

## Exécution

Pour lancer le projet, utilisez directement l'exécutable Python du venv :

```powershell
.\venv\Scripts\python.exe main.py
```

Alternativement, vous pouvez activer l'environnement virtuel puis lancer le script :

```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

## Structure du projet

- `main.py` : Script principal pour entraîner et évaluer le modèle
- `breast_cancer_classifier.py` : Classe du classifieur avec architecture du réseau
- `breast-train.csv` : Données d'entraînement
- `requirements.txt` : Liste des dépendances
