import cv2
import numpy as np
from keras.models import load_model
import time
from collections import deque
import argparse

# ======================
# 1. INITIALIZE MODELS
# ======================

# Improved face detection (DNN)
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",   # Download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    "res10_300x300_ssd_iter_140000.caffemodel"   # Download from: https://github.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
)

# FER2013 Model
fer_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False) # Download from: https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5
fer_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion colors for visualization
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 102, 0),    # Dark Green
    'Fear': (102, 0, 102),     # Purple
    'Happy': (0, 255, 255),    # Yellow
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (0, 255, 0),   # Green
    'Neutral': (200, 200, 200) # Light Gray
}

# ======================
# 2. HELPER FUNCTIONS
# ======================

def process_face_for_fer(face_img):
    """Enhanced preprocessing for FER2013 model with multiple techniques"""
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    
    # Apply CLAHE for adaptive contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur for noise reduction
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Resize with anti-aliasing
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Normalize and expand dimensions
    normalized = resized.astype('float32') / 255.0
    
    return np.expand_dims(normalized, axis=(0, -1))

def detect_faces_dnn(frame, confidence_threshold=0.8):
    """Improved face detection using DNN with better boundary handling"""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            # Ensure coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Only add if we have a valid region
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2-x1, y2-y1))
    
    return faces

def apply_temporal_smoothing(predictions, history_length=5):
    """Apply temporal smoothing to predictions using weighted history"""
    if not hasattr(apply_temporal_smoothing, 'history'):
        apply_temporal_smoothing.history = deque(maxlen=history_length)
    
    apply_temporal_smoothing.history.append(predictions)
    
    # Weight more recent predictions higher
    weights = np.linspace(0.5, 1.0, len(apply_temporal_smoothing.history))
    weights /= weights.sum()
    
    smoothed = np.zeros_like(predictions)
    for i, pred in enumerate(apply_temporal_smoothing.history):
        smoothed += pred * weights[i]
    
    return smoothed

def draw_emotion_chart(frame, emotions, confidences, top_left, size):
    """Draw a bar chart showing emotion probabilities"""
    chart_width, chart_height = size
    x, y = top_left
    
    # Draw chart background
    cv2.rectangle(frame, (x, y), (x + chart_width, y + chart_height), (50, 50, 50), -1)
    
    # Draw bars for each emotion
    bar_width = chart_width // len(emotions)
    max_height = chart_height - 20
    
    for i, (emotion, conf) in enumerate(zip(emotions, confidences)):
        bar_height = int(conf * max_height)
        color = emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw the bar
        cv2.rectangle(frame, 
                     (x + i * bar_width, y + chart_height - bar_height),
                     (x + (i + 1) * bar_width, y + chart_height),
                     color, -1)
        
        # Draw the label
        cv2.putText(frame, emotion[0],  # Just first letter for compactness
                   (x + i * bar_width + 5, y + chart_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def process_frame(frame, emotion_history):
    """Process a single frame for emotion detection"""
    # Mirror the frame for more natural interaction (only for webcam)
    frame = cv2.flip(frame, 1)

    # Detect faces using DNN
    faces = detect_faces_dnn(frame)

    for i, (x, y, w, h) in enumerate(faces):
        # Draw face rectangle with emotion color
        emotion_color = (0, 255, 0)  # Default to green if no emotion detected yet
        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_color, 2)
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Process and predict with FER2013 model
        fer_input = process_face_for_fer(face_roi)
        fer_pred = fer_model.predict(fer_input, verbose=0)[0]
        
        # Apply temporal smoothing
        fer_pred = apply_temporal_smoothing(fer_pred)
        
        # Get top emotion and confidence
        top_idx = np.argmax(fer_pred)
        fer_emotion = fer_emotions[top_idx]
        fer_confidence = fer_pred[top_idx]
        
        # Update emotion history
        emotion_history.append((fer_emotion, fer_confidence))
        
        # Get most confident recent emotion
        if emotion_history:
            fer_emotion, fer_confidence = max(emotion_history, key=lambda x: x[1])
        
        # Get color for the detected emotion
        emotion_color = emotion_colors.get(fer_emotion, (0, 255, 0))
        
        # Update face rectangle with emotion color
        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_color, 2)
        
        # Draw emotion label with confidence
        label = f"{fer_emotion} ({fer_confidence:.0%})"
        cv2.putText(frame, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        
        # Draw emotion chart for the primary face (first detected)
        if i == 0:
            chart_position = (20, 20)
            chart_size = (200, 100)
            draw_emotion_chart(frame, fer_emotions, fer_pred, chart_position, chart_size)
            
            # Display detailed confidence values
            for j, (emotion, conf) in enumerate(zip(fer_emotions, fer_pred)):
                if conf > 0.1:  # Only show significant probabilities
                    cv2.putText(frame, f"{emotion}: {conf:.1%}",
                               (chart_position[0] + chart_size[0] + 10, 
                                chart_position[1] + 15 + j * 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_colors.get(emotion, (255, 255, 255)), 1)
    
    return frame

# ======================
# 3. MAIN FUNCTIONALITY
# ======================

def webcam_mode():
    """Run emotion detection on webcam feed"""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)

    prev_time = 0
    fps_history = deque(maxlen=10)
    emotion_history = deque(maxlen=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)

        # Process frame
        frame = process_frame(frame, emotion_history)

        # Display FPS and instructions
        cv2.rectangle(frame, (10, frame.shape[0] - 60), (250, frame.shape[0] - 10), (40, 40, 40), -1)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('Emotion Detection - Webcam Mode', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def image_mode(image_path):
    """Run emotion detection on a single image"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    emotion_history = deque(maxlen=1)  # No temporal smoothing for single image
    
    # Process the image
    frame = process_frame(frame, emotion_history)
    
    # Display the result
    cv2.imshow('Emotion Detection - Image Mode', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the result
    output_path = "output_" + image_path
    cv2.imwrite(output_path, frame)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion Detection from Webcam or Image')
    parser.add_argument('--image', type=str, help='Path to input image file')
    args = parser.parse_args()

    if args.image:
        image_mode(args.image)
    else:
        webcam_mode()