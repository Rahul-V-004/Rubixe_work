import cv2
import dlib
import numpy as np
from collections import deque

# Load dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
GAZE_THRESHOLD = 0.25
HISTORY_LENGTH = 5

def get_eye_center(landmarks, eye_indices):
    return np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices], axis=0)

def detect_gaze(landmarks):
    left_eye_center = get_eye_center(landmarks, LEFT_EYE_INDICES)
    right_eye_center = get_eye_center(landmarks, RIGHT_EYE_INDICES)
    eyes_center = np.mean([left_eye_center, right_eye_center], axis=0)
    
    nose_bridge = np.array([landmarks.part(30).x, landmarks.part(30).y])
    face_width = landmarks.part(16).x - landmarks.part(0).x
    
    distance = np.linalg.norm(eyes_center - nose_bridge)
    return (distance / face_width) < GAZE_THRESHOLD

def main():
    cap = cv2.VideoCapture(0)
    gaze_history = deque(maxlen=HISTORY_LENGTH)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        looking_at_camera = False
        for face in faces:
            landmarks = predictor(gray, face)
            looking_at_camera = detect_gaze(landmarks)
            gaze_history.append(looking_at_camera)
            
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Smooth the gaze detection using a moving average
        looking_at_camera = sum(gaze_history) > len(gaze_history) // 2
        
        text = "Looking at camera" if looking_at_camera else "Not looking at camera"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()