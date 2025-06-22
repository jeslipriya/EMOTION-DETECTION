import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

# Emotion labels (DeepFace uses these)
emotion_labels = {
    'angry': ' Angry',
    'disgust': ' Disgust',
    'fear': ' Fear',
    'happy': ' Happy',
    'sad': ' Sad',
    'surprise': ' Surprise',
    'neutral': ' Neutral'
}

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze face for emotions (DeepFace handles face detection)
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
        
        # Get the first face detected
        face_data = result[0]
        emotion = face_data['dominant_emotion']
        confidence = face_data['emotion'][emotion]
        
        # Get face location
        x, y, w, h = face_data['region']['x'], face_data['region']['y'], face_data['region']['w'], face_data['region']['h']
        
        # Only show if confidence > 40%
        if confidence > 40:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display emotion text with emoji
            text = f"{emotion_labels[emotion]} ({confidence:.1f}%)"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"Error: {e}")  # Skip if no face is detected

    # Display the webcam feed
    cv2.imshow('Real-Time Emotion Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()