import cv2
import numpy as np
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Emotion mapping (DeepFace uses different labels)
emotion_mapping = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'sad': 'Sad',
    'surprise': 'Surprise',
    'neutral': 'Neutral'
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze frame with DeepFace (more accurate than FER2013 model)
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
        
        for result in results:
            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
            dominant_emotion = result['dominant_emotion']
            emotion_score = result['emotion'][dominant_emotion]
            
            # Only show if confidence > 40%
            if emotion_score > 40:
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Text background for readability
                text = f"{emotion_mapping[dominant_emotion]} {emotion_score:.1f}%"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y-35), (x+text_width+10, y), (0,0,0), -1)
                
                # Put emotion text
                cv2.putText(frame, text, (x+5, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Visual emotion intensity indicator
                bar_length = int(100 * (emotion_score/100))
                cv2.rectangle(frame, (x, y+h+5), (x+bar_length, y+h+15), (0, 255, 0), -1)
    
    except Exception as e:
        print(f"Error: {e}")
        continue

    cv2.imshow('Advanced Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()