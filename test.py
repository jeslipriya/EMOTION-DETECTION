import cv2
import numpy as np
from keras.models import load_model
from deepface import DeepFace
import time

# ======================
# 1. INITIALIZE MODELS
# ======================

# Improved face detection (DNN)
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # Download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    "res10_300x300_ssd_iter_140000.caffemodel"  # Download from: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
)

# FER2013 Model with enhancements
fer_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
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
    
    # Add slight augmentation
    if np.random.rand() > 0.5:
        normalized = np.fliplr(normalized)
    
    return np.expand_dims(normalized, axis=(0, -1))

def analyze_with_deepface(face_img):
    """Enhanced DeepFace analysis with error handling"""
    try:
        # Convert to RGB and enhance lighting
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        limg = cv2.merge((clahe.apply(l), a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        result = DeepFace.analyze(enhanced, actions=['emotion'], enforce_detection=False, silent=True)
        return result[0]['dominant_emotion'], result[0]['emotion']
    except Exception as e:
        print(f"DeepFace error: {str(e)}")
        return "Unknown", {}

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
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows reduces latency
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Higher resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)  # Target 30 FPS
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)  # Adjust brightness

# For FPS calculation
prev_time = 0
fps_history = []

# Emotion history for smoothing
emotion_history = {'FER': [], 'DeepFace': []}
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
        # Expand face ROI slightly
        padding = int(w * 0.1)
        x, y = max(0, x-padding), max(0, y-padding)
        w, h = min(w+2*padding, frame.shape[1]-x), min(h+2*padding, frame.shape[0]-y)
        
        # Draw face rectangle with gradient effect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y), (0, 255, 0), -1)
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # ==================================
        # A. PREDICT WITH FER2013 MODEL (ENHANCED)
        # ==================================
        fer_input = process_face_for_fer(face_roi)
        fer_pred = fer_model.predict(fer_input, verbose=0)[0]
        
        # Apply softmax temperature scaling
        temperature = 0.7
        scaled_pred = np.exp(np.log(fer_pred + 1e-10) / temperature)
        fer_pred = scaled_pred / np.sum(scaled_pred)
        
        fer_emotion = fer_emotions[np.argmax(fer_pred)]
        fer_confidence = np.max(fer_pred)
        
        # Smooth predictions using history
        emotion_history['FER'].append((fer_emotion, fer_confidence))
        if len(emotion_history['FER']) > HISTORY_LENGTH:
            emotion_history['FER'].pop(0)
        
        # Get most frequent recent emotion
        if emotion_history['FER']:
            emotions, confidences = zip(*emotion_history['FER'])
            fer_emotion = max(set(emotions), key=emotions.count)
            fer_confidence = np.mean([c for e, c in emotion_history['FER'] if e == fer_emotion])
        
        # Display FER2013 result
        cv2.putText(frame, f"FER: {fer_emotion} ({fer_confidence:.1%})", 
                   (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
        
        # ==================================
        # B. PREDICT WITH DEEPFACE (ENHANCED)
        # ==================================
        df_emotion, df_all_emotions = analyze_with_deepface(face_roi)
        df_confidence = df_all_emotions.get(df_emotion, 0) / 100
        
        # Smooth DeepFace predictions
        emotion_history['DeepFace'].append((df_emotion, df_confidence))
        if len(emotion_history['DeepFace']) > HISTORY_LENGTH:
            emotion_history['DeepFace'].pop(0)
        
        if emotion_history['DeepFace']:
            emotions, confidences = zip(*emotion_history['DeepFace'])
            df_emotion = max(set(emotions), key=emotions.count)
            df_confidence = np.mean([c for e, c in emotion_history['DeepFace'] if e == df_emotion])
        
        # Display DeepFace result with emoji
        emoji_map = {
            'angry': '😠', 'disgust': '🤢', 'fear': '😨',
            'happy': '😊', 'sad': '😢', 'surprise': '😮',
            'neutral': '😐'
        }
        emoji = emoji_map.get(df_emotion.lower(), '')
        cv2.putText(frame, f"DeepFace: {emoji}{df_emotion} ({df_confidence:.1%})", 
                   (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        
        # Draw confidence bar
        bar_length = int(w * df_confidence)
        cv2.rectangle(frame, (x, y+h+5), (x+bar_length, y+h+15), (200, 200, 0), -1)

    # Display FPS and instructions
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press Q to quit", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Enhanced Emotion Detection', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()