import cv2
import numpy as np
from keras.models import load_model
from deepface import DeepFace

# ======================
# 1. INITIALIZE MODELS
# ======================

# Face detection (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# FER2013 Model (your current model)
fer_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
fer_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ======================
# 2. HELPER FUNCTIONS
# ======================

def process_face_for_fer(face_img):
    """Preprocess face image for FER2013 model"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

def analyze_with_deepface(face_img):
    """Get emotion using DeepFace"""
    try:
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
        return result[0]['dominant_emotion'], result[0]['emotion']
    except:
        return "Unknown", {}

# ======================
# 3. MAIN LOOP
# ======================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # ==================================
        # A. PREDICT WITH FER2013 MODEL
        # ==================================
        fer_input = process_face_for_fer(face_roi)
        fer_pred = fer_model.predict(fer_input, verbose=0)[0]
        fer_emotion = fer_emotions[np.argmax(fer_pred)]
        fer_confidence = np.max(fer_pred)
        
        # Display FER2013 result (left side)
        cv2.putText(frame, f"FER: {fer_emotion} ({fer_confidence:.1%})", 
                   (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
        
        # ==================================
        # B. PREDICT WITH DEEPFACE
        # ==================================
        df_emotion, df_all_emotions = analyze_with_deepface(face_roi)
        df_confidence = df_all_emotions.get(df_emotion, 0) / 100
        
        # Display DeepFace result (right side)
        cv2.putText(frame, f"DeepFace: {df_emotion} ({df_confidence:.1%})", 
                   (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    # Show frame
    cv2.imshow('Emotion Detection Comparison', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()