import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import build_dataset, SEQUENCE_LENGTH, IMG_SIZE
from model import build_cnn_model, build_cnn_lstm_model


def make_ds(X, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main(dataset_dir, baseline=False, epochs=25, batch_size=4):
    X, y, class_names = build_dataset(dataset_dir, SEQUENCE_LENGTH, IMG_SIZE)
    if len(X) == 0:
        print("Dataset is empty.")
        return
    # 70% train / 10% validation / 20% test (stratified)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.125, random_state=42, stratify=y_tmp)
    print("Train/Val/Test:", len(X_train), len(X_val), len(X_test))

    mid = SEQUENCE_LENGTH // 2
    if baseline:
        # Baseline CNN sees one frame (the middle frame) per clip
        X_train, X_val, X_test = X_train[:, mid], X_val[:, mid], X_test[:, mid]
        model = build_cnn_model(IMG_SIZE, len(class_names))
        tag = "baseline"
    else:
        model = build_cnn_lstm_model(
            SEQUENCE_LENGTH, IMG_SIZE, len(class_names))
        tag = "cnn_lstm"
    history = model.fit(
        make_ds(X_train, y_train, batch_size, shuffle=True),
        validation_data=make_ds(X_val, y_val, batch_size),
        epochs=epochs
    )
    model.save(f"{tag}_model.keras")
    with open("class_names.json", "w") as file:
        json.dump(class_names, file)

    # Final evaluation on the held-out test set only
    test_ds = make_ds(X_test, y_test, batch_size)
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(
        y_test, y_pred, labels=range(len(class_names)),
        target_names=class_names, zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))

    # Figures: training curves and confusion matrix
    plt.figure(figsize=(10, 4))
    for i, m in enumerate(["accuracy", "loss"], start=1):
        plt.subplot(1, 2, i)
        plt.plot(history.history[m], label="train")
        plt.plot(history.history["val_" + m], label="validation")
        plt.title(m.capitalize())
        plt.xlabel("Epoch")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{tag}_training_curves.png", dpi=200)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    for (r, c), v in np.ndenumerate(cm):
        plt.text(c, r, v, ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{tag}_confusion_matrix.png", dpi=200)

    # Figure: sample test predictions (middle frame of each sequence)
    n = min(6, len(X_test))
    plt.figure(figsize=(12, 2.5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X_test[i] if baseline else X_test[i][mid])
        plt.axis("off")
        plt.title(f"T: {class_names[y_test[i]]}\nP: {class_names[y_pred[i]]}",
                  fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{tag}_sample_predictions.png", dpi=200)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <dataset_folder> [--baseline]")
        sys.exit(1)
    main(sys.argv[1], baseline="--baseline" in sys.argv)
