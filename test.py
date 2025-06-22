import cv2
import numpy as np
from keras.models import load_model
import time

# ======================
# 1. INITIALIZE MODELS
# ======================

# Improved face detection (DNN)
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # Download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    "res10_300x300_ssd_iter_140000.caffemodel"  # Download from: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
)

# FER2013 Model
fer_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)   # Download from: https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5
fer_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ======================
# 2. HELPER FUNCTIONS
# ======================

def process_face_for_fer(face_img):
    """Enhanced preprocessing for FER2013 model"""
    # Convert to grayscale with CLAHE for better contrast
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Resize with anti-aliasing
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype('float32') / 255.0
    
    return np.expand_dims(normalized, axis=(0, -1))

def detect_faces_dnn(frame):
    """Improved face detection using DNN"""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.7:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2-x1, y2-y1))
    
    return faces

# ======================
# 3. MAIN LOOP
# ======================

# Initialize video with better settings
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)

# For FPS calculation
prev_time = 0
fps_history = []

# Emotion history for smoothing
emotion_history = []
HISTORY_LENGTH = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_history.append(fps)
    if len(fps_history) > 10:
        fps_history.pop(0)
    avg_fps = sum(fps_history) / len(fps_history)

    # Mirror the frame for more natural interaction
    frame = cv2.flip(frame, 1)

    # Detect faces using DNN
    faces = detect_faces_dnn(frame)

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face ROI (no zooming)
        face_roi = frame[y:y+h, x:x+w]
        
        # Process and predict with FER2013 model
        fer_input = process_face_for_fer(face_roi)
        fer_pred = fer_model.predict(fer_input, verbose=0)[0]
        
        # Apply softmax temperature scaling
        temperature = 0.7
        scaled_pred = np.exp(np.log(fer_pred + 1e-10) / temperature)
        fer_pred = scaled_pred / np.sum(scaled_pred)
        
        fer_emotion = fer_emotions[np.argmax(fer_pred)]
        fer_confidence = np.max(fer_pred)
        
        # Smooth predictions using history
        emotion_history.append((fer_emotion, fer_confidence))
        if len(emotion_history) > HISTORY_LENGTH:
            emotion_history.pop(0)
        
        # Get most frequent recent emotion
        if emotion_history:
            emotions, confidences = zip(*emotion_history)
            fer_emotion = max(set(emotions), key=emotions.count)
            fer_confidence = np.mean([c for e, c in emotion_history if e == fer_emotion])
        
        # Display result with colored confidence
        color = (0, 255, 0)  # Green
        if fer_confidence < 0.5:
            color = (0, 165, 255)  # Orange
        if fer_confidence < 0.3:
            color = (0, 0, 255)  # Red
            
        cv2.putText(frame, f"{fer_emotion} ({fer_confidence:.1%})", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display FPS and instructions
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press Q to quit", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('High Quality Emotion Detection', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()