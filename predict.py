import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import frames_from_video, SEQUENCE_LENGTH, IMG_SIZE


def predict_video(video_path,
                  model_path="cnn_lstm_model.keras",
                  classes_path="class_names.json"):
    model = load_model(model_path)
    with open(classes_path, "r") as file:
        class_names = json.load(file)
    frames = frames_from_video(video_path, SEQUENCE_LENGTH, IMG_SIZE)
    if frames is None:
        print("Unable to read video.")
        return
    predictions = model.predict(np.expand_dims(frames, axis=0), verbose=0)[0]
    top = np.argsort(predictions)[::-1][:3]
    print("\nTop Predictions")
    print("-" * 35)
    for rank, index in enumerate(top, start=1):
        print(f"{rank}. {class_names[index]:20s}"
              f"{predictions[index]*100:.2f}%")
    print("\nPredicted Activity:", class_names[top[0]])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <video_path>")
        sys.exit(1)
    predict_video(sys.argv[1])
