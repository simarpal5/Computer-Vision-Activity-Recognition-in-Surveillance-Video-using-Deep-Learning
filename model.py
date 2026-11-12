from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, TimeDistributed,
    GlobalAveragePooling2D, BatchNormalization,
    LSTM, Dense, Dropout
)


def build_cnn_model(img_size, num_classes):
    """Baseline: classifies a single frame (no temporal modelling)."""
    model = Sequential(name="Baseline_CNN")
    model.add(Input(shape=(img_size, img_size, 3)))
    for filters in (32, 64, 128):
        model.add(Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_cnn_lstm_model(sequence_length, img_size, num_classes):
    model = Sequential(name="CNN_LSTM_GAP")
    model.add(Input(shape=(sequence_length, img_size, img_size, 3)))
    for filters in (32, 64, 128):
        model.add(TimeDistributed(
            Conv2D(filters, (3, 3), activation="relu", padding="same")))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    # Three classes: fighting, loitering, normal
    model = build_cnn_lstm_model(
        sequence_length=20, img_size=64, num_classes=3)
    model.summary()
