import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Improved emotion model (consider using DeepFace or FER+ models)
emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# More accurate emotion labels (merged some similar emotions)
emotion_labels = ['Angry', 'Disgust/Fear', 'Disgust/Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Use more accurate face detector (DNN or MTCNN would be better)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam with higher resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Parameters for better face detection
min_confidence = 0.5  # Minimum confidence to display emotion

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale and equalize histogram for better contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Detect faces with more sensitive parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        # Extract face ROI with padding
        padding = 30
        y1 = max(0, y - padding)
        y2 = min(frame.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(frame.shape[1], x + w + padding)
        
        face_roi = gray[y1:y2, x1:x2]
        
        try:
            # Resize and preprocess with more sophisticated approach
            roi = cv2.resize(face_roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Predict emotion with timing for performance measurement
            preds = emotion_model.predict(roi, verbose=0)[0]
            emotion_probability = np.max(preds)
            label = emotion_labels[np.argmax(preds)]
            
            # Only display if confidence is high enough
            if emotion_probability > min_confidence:
                # Draw rectangle and label with more visible styling
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
                
                # Create background for text for better visibility
                text_width, text_height = cv2.getTextSize(
                    f"{label} {emotion_probability*100:.1f}%",
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                cv2.rectangle(frame, 
                              (x, y - text_height - 15),
                              (x + text_width + 10, y),
                              (0, 0, 0), -1)
                
                cv2.putText(frame, 
                          f"{label} {emotion_probability*100:.1f}%", 
                          (x + 5, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                          (0, 255, 255), 2)
                
                # Add face landmarks or other visual cues
                cv2.circle(frame, (x + w//2, y + h//2), 3, (0, 0, 255), -1)
        
        except Exception as e:
            print(f"Error processing face: {e}")
            continue
    
    # Add FPS counter
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Improved Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()