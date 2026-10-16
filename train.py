"""Train the Climate Vision CNN and evaluate it on a held-out test set.

    python train.py                     # uses data/Multi-class Weather Dataset
    python train.py --epochs 30 --seed 7

Writes models/weather_cnn.keras, models/class_names.json, and the evaluation
reports in reports/ (metrics.json, confusion_matrix.png, training_curves.png).
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are written to files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from preprocess import IMAGE_SIZE, load_image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_dataset(data_dir):
    """Load every image into memory. 1,125 photos at 224x224 is ~170 MB."""
    class_names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    images, labels, skipped = [], [], []
    for label, name in enumerate(class_names):
        folder = os.path.join(data_dir, name)
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            image = load_image(os.path.join(folder, filename))
            if image is None:
                skipped.append(os.path.join(name, filename))
                continue
            images.append(image)
            labels.append(label)
    return np.stack(images), np.array(labels), class_names, skipped


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        # Augmentation runs only during training; it is inactive at inference.
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        # Scaling lives inside the model, so the app feeds raw 0-255 pixels
        # and can't forget (or double-apply) it.
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # Global average pooling instead of Flatten: Flatten -> Dense(256) on a
        # 14x14x256 map is 12.8M weights, most of the model's size and most of
        # its tendency to memorise a 1,125-image dataset.
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        # Softmax, not sigmoid: the four classes are mutually exclusive, so the
        # outputs must be one probability distribution that sums to 1.
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        # from_logits=False because the last layer already applies softmax.
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def save_curves(history, path):
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4))
    loss_ax.plot(history.history["loss"], label="train")
    loss_ax.plot(history.history["val_loss"], label="validation")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("epoch")
    loss_ax.legend()
    acc_ax.plot(history.history["accuracy"], label="train")
    acc_ax.plot(history.history["val_accuracy"], label="validation")
    acc_ax.set_title("Accuracy")
    acc_ax.set_xlabel("epoch")
    acc_ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, class_names, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=class_names
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.path.join("data", "Multi-class Weather Dataset"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    x, y, class_names, skipped = load_dataset(args.data_dir)
    print(f"Loaded {len(x)} images across {class_names}; skipped {len(skipped)} unreadable")

    # 70/15/15, stratified so every split keeps the class proportions. The test
    # split is never seen during training or early stopping, so its score is
    # an honest estimate.
    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=args.seed)
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=args.seed)
    print(f"train {len(x_train)} / validation {len(x_val)} / test {len(x_test)}")

    model = build_model(len(class_names))
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True)],
        verbose=2,
    )

    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    metrics = {
        "test_accuracy": round(float(test_accuracy), 4),
        "test_loss": round(float(test_loss), 4),
        "best_epoch": best_epoch,
        "epochs_run": len(history.history["loss"]),
        "validation_accuracy_at_best_epoch": round(
            float(history.history["val_accuracy"][best_epoch - 1]), 4),
        "split": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "seed": args.seed,
        "per_class": {
            name: {k: round(report[name][k], 4) for k in ("precision", "recall", "f1-score")}
            | {"support": int(report[name]["support"])}
            for name in class_names
        },
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
    }
    with open(os.path.join("reports", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_curves(history, os.path.join("reports", "training_curves.png"))
    save_confusion_matrix(y_test, y_pred, class_names,
                          os.path.join("reports", "confusion_matrix.png"))

    model.save(os.path.join("models", "weather_cnn.keras"))
    with open(os.path.join("models", "class_names.json"), "w") as f:
        json.dump(class_names, f)
    print(f"Test accuracy {test_accuracy:.4f}; model saved to models/weather_cnn.keras")


if __name__ == "__main__":
    main()
