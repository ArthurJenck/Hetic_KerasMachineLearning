from breast_cancer_classifier import BreastCancerClassifier
import numpy as np

# Hyperparamètres
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Initialisation du classifieur
classifier = BreastCancerClassifier(learning_rate=LEARNING_RATE)

# Chargement des données
X, y = classifier.load_data("breast-train.csv")

# Préparation des données (division + normalisation)
X_train, X_test, y_train, y_test = classifier.prepare_data(X, y, test_size=0.2, random_state=42)

# Construction du modèle
classifier.build_model(input_shape=X_train.shape[1])
classifier.model.summary()

# Entraînement
print(f"\n Entraînement du modèle (sur {NUM_EPOCHS} époques)...")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Validation split: {VALIDATION_SPLIT}")

history = classifier.train(
    X_train, y_train,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=1
)

# Sauvegarde du modèle
classifier.save("breast_cancer_final_model.keras")
print("   Modèle sauvegardé: breast_cancer_final_model.keras")

# Évaluation sur le test set
test_loss, test_accuracy, test_precision, test_recall = classifier.evaluate(X_test, y_test)
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_accuracy:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall: {test_recall:.4f}")

# Prédictions
y_pred_proba = classifier.predict(X_test).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

# Rapport de classification détaillé
print("\n Rapport de classification détaillé:")
classifier.print_classification_report(y_test, y_pred)

# Visualisations
print("\n Génération des visualisations...")
print("    - Courbes d'entraînement (loss, accuracy, precision, recall)")
classifier.plot_training_history()

print("    - Matrice de confusion")
classifier.plot_confusion_matrix(y_test, y_pred)

print("    - Courbe ROC")
classifier.plot_roc_curve(y_test, y_pred_proba)

print("\n" + "=" * 60)