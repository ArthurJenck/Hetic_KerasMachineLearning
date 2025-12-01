import csv
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras

class BreastCancerClassifier:
    def __init__(self, hidden_units=256, dropout_rate=0.3, learning_rate=1e-2):
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.mean = None
        self.std = None
        self.class_weight = None

    def load_data(self, filename):
        all_features = []
        all_targets = []

        with open(filename) as file:
            reader = csv.reader(file)

            next(reader)

            for i, row in enumerate(reader):
                features_row = [float(val) for val in row[:-1]]
                all_features.append(features_row)
                all_targets.append([int(float(row[-1]))])

        features = np.array(all_features, dtype="float32")
        targets = np.array(all_targets, dtype="uint8")

        return features, targets

    def prepare_datasets(self, features, targets, validation_split=0.2):
        num_val_samples = int(len(features) * validation_split)

        train_features = features[:-num_val_samples]
        train_targets = targets[:-num_val_samples]
        val_features = features[-num_val_samples:]
        val_targets = targets[-num_val_samples:]

        return train_features, train_targets, val_features, val_targets

    def calculate_class_weights(self, targets):
        counts = np.bincount(targets[:, 0])

        weight_for_0 = 1.0 / counts[0]
        weight_for_1 = 1.0 / counts[1]

        self.class_weight = {0: weight_for_0, 1: weight_for_1}
        return self.class_weight

    def normalize(self, train_features, val_features):
        self.mean = np.mean(train_features, axis=0)
        self.std = np.std(train_features, axis=0)

        train_features_normalized = (train_features - self.mean) / self.std
        val_features_normalized = (val_features - self.mean) / self.std

        return train_features_normalized, val_features_normalized

    def build_model(self, input_shape):
        self.model = keras.Sequential([
            keras.Input(shape=input_shape),
            keras.layers.Dense(self.hidden_units, activation="relu"),
            keras.layers.Dense(self.hidden_units,activation="relu"),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(self.hidden_units, activation="relu"),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.FalseNegatives(name="fn"),
                keras.metrics.FalsePositives(name="fp"),
                keras.metrics.TrueNegatives(name="tn"),
                keras.metrics.TruePositives(name="tp"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        return self.model

    def train(self, train_features, train_targets, val_features, val_targets, epochs=30, batch_size=2048, model_prefix="breast_model"):
        callbacks = [
            keras.callbacks.ModelCheckpoint(f"{model_prefix}_epoch_{{epoch}}.keras")
        ]

        history = self.model.fit(
            train_features,
            train_targets,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
            validation_data=(val_features, val_targets),
            class_weight=self.class_weight,
        )

        return history

    def predict(self, features):
        if self.mean is not None and self.std is not None:
            features_normalized = (features - self.mean) / self.std
        else:
            features_normalized = features

        return self.model.predict(features_normalized)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)