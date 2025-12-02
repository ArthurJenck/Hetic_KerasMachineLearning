import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


class BreastCancerClassifier:
    def __init__(self, learning_rate=0.001):
        """
        Initialisation du classifieur avec les hyperparamètres.
        """
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def load_data(self, filename):
        """
        Charge les données depuis un fichier CSV.
        
        Args:
            filename: Chemin vers le fichier CSV
            
        Returns:
            X: Features (numpy array)
            y: Targets (numpy array)
        """
        all_features = []
        all_targets = []
        
        with open(filename) as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            
            for row in reader:
                features_row = [float(val) for val in row[:-1]]
                all_features.append(features_row)
                all_targets.append(int(float(row[-1])))
        
        X = np.array(all_features, dtype="float32")
        y = np.array(all_targets, dtype="int32")
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Division et normalisation des données.
        
        Args:
            X: Features
            y: Targets
            test_size: Proportion du test set
            random_state: Seed pour la reproductibilité
            
        Returns:
            X_train, X_test, y_train, y_test normalisés
        """
        # Division train/test avec stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalisation (StandardScaler)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """
        Construction du modèle.
        
        Args:
            input_shape: Nombre de features en entrée
        """
        self.model = Sequential([
            Input(shape=(input_shape,)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1):
        """
        Entraînement du modèle.
        
        Args:
            X_train: Features d'entraînement
            y_train: Targets d'entraînement
            epochs: Nombre d'époques
            batch_size: Taille des batchs
            validation_split: Proportion de validation
            verbose: Niveau de verbosité
            
        Returns:
            history: Historique d'entraînement
        """
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        Prédictions sur de nouvelles données.
        
        Args:
            X: Features (déjà normalisées ou non)
            
        Returns:
            y_pred_proba: Probabilités prédites
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Évaluation du modèle sur le test set.
        
        Args:
            X_test: Features de test
            y_test: Targets de test
            
        Returns:
            Métriques d'évaluation
        """
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def plot_training_history(self):
        """
        Affiche les courbes d'entraînement:
        - Loss et val_loss
        - Accuracy et val_accuracy
        - Recall et val_recall
        - Precision et val_precision
        """
        if self.history is None:
            print("Aucun historique d'entraînement disponible. Entraînez d'abord le modèle.")
            return
        
        history_dict = self.history.history
        
        plt.figure(figsize=(12, 10))
        
        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(history_dict['loss'], label='Perte sur entraînement')
        plt.plot(history_dict['val_loss'], label='Perte sur validation')
        plt.title('Évolution de la perte (loss)')
        plt.xlabel('Époque')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        
        # Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history_dict['accuracy'], label='Accuracy entraînement')
        plt.plot(history_dict['val_accuracy'], label='Accuracy validation')
        plt.title('Évolution de la justesse (accuracy)')
        plt.xlabel('Époque')
        plt.ylabel('Justesse')
        plt.legend()
        plt.grid()
        
        # Recall
        plt.subplot(2, 2, 3)
        plt.plot(history_dict['recall'], label='Recall entraînement')
        plt.plot(history_dict['val_recall'], label='Recall validation')
        plt.title('Évolution du rappel (recall)')
        plt.xlabel('Époque')
        plt.ylabel('Rappel')
        plt.legend()
        plt.grid()
        
        # Precision
        plt.subplot(2, 2, 4)
        plt.plot(history_dict['precision'], label='Precision entraînement')
        plt.plot(history_dict['val_precision'], label='Precision validation')
        plt.title('Évolution de la précision (precision)')
        plt.xlabel('Époque')
        plt.ylabel('Précision')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Affiche la matrice de confusion.
        
        Args:
            y_test: Vraies étiquettes
            y_pred: Prédictions (classes 0/1)
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matrice de confusion')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Malin', 'Bénin'])
        plt.yticks(tick_marks, ['Malin', 'Bénin'])
        plt.ylabel('Vrai label')
        plt.xlabel('Label prédit')
        
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='white')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """
        Affiche la courbe ROC.
        
        Args:
            y_test: Vraies étiquettes
            y_pred_proba: Probabilités prédites
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificateur aléatoire')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Courbe ROC - Cancer du sein')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def print_classification_report(self, y_test, y_pred):
        """
        Affiche le rapport de classification.
        
        Args:
            y_test: Vraies étiquettes
            y_pred: Prédictions (classes 0/1)
        """
        print("Rapport de classification :")
        print(classification_report(y_test, y_pred, target_names=['Malin', 'Bénin']))
    
    def save(self, filepath):
        """Sauvegarde le modèle."""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Charge un modèle sauvegardé."""
        self.model = load_model(filepath)
