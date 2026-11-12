**Human Activity Recognition in Surveillance Videos Using Deep Learning**
**Overview**
This project focuses on developing an AI-powered Human Activity Recognition (HAR) system using Deep Learning and Computer Vision. It analyzes surveillance videos to identify and classify human activities using a CNN-LSTM architecture.
CNN extracts spatial features from individual video frames, while LSTM captures temporal patterns across sequences to understand activities over time.

**Objectives**
- Automate human activity recognition in surveillance videos.
- Identify suspicious activities such as fighting and abnormal behaviour.
- Explore deep learning techniques for video analysis.
- Investigate challenges such as occlusion and low-resolution footage.

**Model Architecture**
The project uses a hybrid CNN-LSTM architecture:
- CNN: Extracts spatial features from video frames.
- LSTM: Learns temporal dependencies between consecutive frames.
- Classification Layer: Predicts the activity category.

**Tech Stack**
- Python
- TensorFlow & Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- YOLOv8 (computer vision exploration)

**Project Structure**
```text
Human-Activity-Recognition/
│
├── dataset/
├── models/
├── preprocess.py
├── model.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

**Installation**

1. Clone the repository:
```bash
git clone YOUR_REPOSITORY_URL
cd Human-Activity-Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python train.py
```

4. Run activity prediction:
```bash
python predict.py
```

**Research Focus**
The project explores the application of deep learning in intelligent surveillance, focusing on spatial-temporal feature extraction, activity classification, and challenges associated with real-world video analysis.

**Future Scope**
- Real-time activity recognition.
- Improved accuracy under occlusion and poor lighting.
- Edge-device deployment.
- Integration with intelligent surveillance applications.

**Author**
Simar Pal Singh 
B.Tech – Computer Science Engineering  
Amity University, Lucknow

---

If you find this project interesting, feel free to explore the repository and contribute!
