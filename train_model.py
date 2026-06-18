"""
train_model.py

Trains a CNN to classify satellite image patches as 'ship' or 'no-ship'.

Dataset:
    Kaggle - "Ships in Satellite Imagery"
    https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery

    Download `shipsnet.json` (or the `shipsnet` folder of labeled PNGs)
    and place it inside the `data/` folder before running this script.

Usage:
    python train_model.py --data data/shipsnet.json --epochs 15
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

IMAGE_SIZE = 80  # shipsnet images are 80x80 RGB
NUM_CHANNELS = 3


def load_dataset(json_path: str):
    """Loads the shipsnet.json dataset into numpy arrays."""
    with open(json_path, "r") as f:
        dataset = json.load(f)

    data = np.array(dataset["data"]).astype("uint8")
    labels = np.array(dataset["labels"]).astype("uint8")

    # Each flattened sample is 3 channels of 80x80 pixels stacked together
    data = data.reshape(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    data = data.transpose(0, 2, 3, 1)  # -> (N, H, W, C)
    data = data.astype("float32") / 255.0

    return data, labels


def build_model():
    """A small CNN — enough capacity for this task without being heavy to train."""
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),  # binary: ship vs no-ship
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ship/no-ship CNN classifier")
    parser.add_argument("--data", default="data/shipsnet.json", help="Path to shipsnet.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="model/ship_detector.h5", help="Where to save the trained model")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Could not find '{args.data}'. Download shipsnet.json from "
            "https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery "
            "and place it in the data/ folder."
        )

    print("Loading dataset...")
    X, y = load_dataset(args.data)
    print(f"Loaded {len(X)} images — {int(y.sum())} ships, {int(len(y) - y.sum())} no-ships")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {accuracy:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
