# 😃 High Quality Real-time Emotion Detection

This project uses OpenCV's DNN face detector and the mini-XCEPTION model trained on FER2013 to perform real-time facial emotion recognition with enhanced accuracy, smoothing, and visualization.

---

## ✨ Features

* 🚀 Real-time emotion detection via webcam
* 💡 DNN-based face detection (more accurate than Haar)
* 🎭 Emotion recognition using FER2013 model
* 🎨 Visual confidence color coding (Green / Orange / Red)
* 🔁 Prediction smoothing using history buffer
* 📊 FPS display + enhanced preprocessing (CLAHE)

---

## 🧐 Emotion Classes

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

## 🛠 Requirements

Install dependencies using pip:

```bash
pip install opencv-python numpy keras tensorflow
```

Also, download the required files and place them in the same directory as your Python script:

| File                                       | Description               | Download Link                                                                                                                                              |
| ------------------------------------------ | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deploy.prototxt`                          | DNN face detector config  | [Download](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)                                               |
| `res10_300x300_ssd_iter_140000.caffemodel` | DNN face detector weights | [Download](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel) |
| `fer2013_mini_XCEPTION.102-0.66.hdf5`      | Pretrained emotion model  | [Download](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5)                  |

---

## ▶️ How to Run

```bash
python your_script_name.py
```

* The webcam will open.
* Make facial expressions (e.g., 😡 😄 😱 😢).
* Press `Q` to quit.

---

## 📸 Output Preview

* \[Face Box] 😄 Happy (94.2%)
* \[Visual Bar] 📊 Emotion intensity
* \[FPS Counter] ⏱️ FPS: 29.7

---

## 🔬 Behind the Scenes

* Face detection using OpenCV's DNN (Caffe SSD)
* Emotion recognition via Keras mini-XCEPTION model
* Softmax temperature scaling for sharper outputs
* Smoothing using last 5 emotion predictions
* Image enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)

---

## 💬 Credits

Built with ❤️ by **Jesli**
