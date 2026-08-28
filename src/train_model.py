"""Train an LSTM classifier from landmarks collected in ``keypoints/``."""

from __future__ import annotations

import argparse
import json

import tensorflow as tf

from config import FEATURE_SIZE, KEYPOINTS_DIR, LABELS_PATH, MODEL_PATH, MODELS_DIR, PHRASE_VOCABULARY, SEQUENCE_LENGTH
from data_loader import load_dataset


def build_model(class_count: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(class_count, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ISL gesture classifier.")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    features, targets, labels = load_dataset(KEYPOINTS_DIR, PHRASE_VOCABULARY)
    if len(labels) < 2:
        raise SystemExit("Collect at least two phrases from the configured banking/hospital vocabulary before training.")
    if len(features) < len(labels) * 2:
        raise SystemExit("Collect at least two complete sequences for each label before training.")

    tf.keras.utils.set_random_seed(42)
    model = build_model(len(labels))
    validation_split = 0.2 if len(features) >= 10 else 0.0
    model.fit(features, targets, epochs=args.epochs, validation_split=validation_split, shuffle=True)

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Labels: {', '.join(labels)}")


if __name__ == "__main__":
    main()
