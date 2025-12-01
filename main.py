from breast_cancer_classifier import BreastCancerClassifier

classifier = BreastCancerClassifier(
    hidden_units=256,
    dropout_rate=0.3,
    learning_rate=1e-2
)

features, targets = classifier.load_data("breast-train.csv")

train_features, train_targets, val_features, val_targets = \
    classifier.prepare_datasets(features, targets)

classifier.calculate_class_weights(train_targets)

train_features_norm, val_features_norm = classifier.normalize(
    train_features, val_features
)

classifier.build_model(train_features_norm.shape[1:])

history = classifier.train(
    train_features_norm,
    train_targets,
    val_features_norm,
    val_targets,
    epochs=30,
    batch_size=2048
)

classifier.save("breast_cancer_final_model.keras")