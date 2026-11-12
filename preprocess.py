import os
import sys
import cv2
import numpy as np

IMG_SIZE = 64
SEQUENCE_LENGTH = 20
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")


def frames_from_video(video_path,
                      sequence_length=SEQUENCE_LENGTH,
                      img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    interval = max(total_frames // sequence_length, 1)
    frames = []
    for i in range(sequence_length):
        frame_number = min(i * interval, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    if len(frames) == 0:
        return None
    while len(frames) < sequence_length:
        frames.append(frames[-1])
    return np.array(frames, dtype=np.float32)


def build_dataset(dataset_dir,
                  sequence_length=SEQUENCE_LENGTH,
                  img_size=IMG_SIZE):
    classes = sorted(
        folder for folder in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, folder))
    )
    X, y = [], []
    for label, cls in enumerate(classes):
        class_folder = os.path.join(dataset_dir, cls)
        for file in sorted(os.listdir(class_folder)):
            if file.lower().endswith(VIDEO_EXTENSIONS):
                path = os.path.join(class_folder, file)
                frames = frames_from_video(path, sequence_length, img_size)
                if frames is not None:
                    X.append(frames)
                    y.append(label)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
            classes)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        X, y, classes = build_dataset(sys.argv[1])
        print("Dataset shape :", X.shape)
        print("Labels shape  :", y.shape)
        print("Classes       :", classes)
    else:
        print("Usage: python preprocess.py <dataset_path>")
