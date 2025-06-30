# Enhanced Real-Time Emotion Detection System

## Overview

This project presents an advanced real-time facial emotion detection system leveraging OpenCV's deep learning-based face detection and a pre-trained mini-XCEPTION model trained on the FER2013 dataset. It is designed to provide high-accuracy emotion recognition with enhanced preprocessing, confidence-based visualization, and temporal smoothing for more stable predictions.

---

## Features

* 🚀 **Real-Time Emotion Detection** via webcam
* 💡 **Accurate Face Detection** using OpenCV DNN (Caffe SSD)
* 🎭 **Emotion Classification** using mini-XCEPTION (FER2013)
* 📊 **Dynamic FPS Counter** for performance feedback
* 🎨 **Color-Coded Emotion Visualization** for clarity
* ⚖️ **Temporal Prediction Smoothing** using weighted history
* 🌐 **Emotion Confidence Chart** for real-time analytics
* 💧 **Enhanced Preprocessing** using CLAHE, Gaussian Blur, Histogram Equalization

---

## Emotion Categories

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

## Requirements

Install the necessary dependencies:

```bash
pip install opencv-python numpy keras tensorflow
```

Ensure the following pre-trained models are downloaded and placed in the same directory:

| File Name                                  | Description               | Download Source                                                                                                                              |
| ------------------------------------------ | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `deploy.prototxt`                          | Face detector config      | [Link](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)                                               |
| `res10_300x300_ssd_iter_140000.caffemodel` | Face detector weights     | [Link](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel) |
| `fer2013_mini_XCEPTION.102-0.66.hdf5`      | Pre-trained emotion model | [Link](https://github.com/oarriaga/face_classification)                                                                                      |

---

## How to Run

```bash
python your_script_name.py
```

* The webcam feed will start.
* Make facial expressions such as happy 😄, angry 😡, sad 😞.
* Press `Q` to quit the application.

---

## Output Display

* Detected faces with colored bounding boxes based on predicted emotion
* Emotion labels with confidence percentages
* Emotion confidence bar chart for the primary face
* FPS counter and quit instructions

---

## Technical Enhancements

* **Face Detection**: Uses OpenCV DNN with Caffe SSD for fast and accurate detection
* **Image Preprocessing**: Combines CLAHE, histogram equalization, and Gaussian blur for improved model input quality
* **Temporal Smoothing**: A deque-based weighted history buffer stabilizes real-time predictions
* **Visualization**: Live emotion probability bar chart and dynamic label rendering improve usability and feedback

---

## Applications

* Human-computer interaction
* Real-time feedback systems
* Affective computing
* Educational or training tools for emotional intelligence

---

## Credits

Developed by Jesli, inspired by the open-source contributions from the computer vision and AI community. Special thanks to the authors of the FER2013 dataset and the mini-XCEPTION model.
