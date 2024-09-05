import cv2
import dlib
import numpy as np
from collections import deque
from scipy.spatial import distance as dist

# Load dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
GAZE_THRESHOLD = 0.25
HISTORY_LENGTH = 5
EYE_AR_THRESH = 0.2  # Eye Aspect Ratio threshold

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_center(landmarks, eye_indices):
    return np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices], axis=0)

def detect_gaze(landmarks):
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES])
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    
    if avg_ear < EYE_AR_THRESH:
        return False, "Eyes Closed"
    
    left_eye_center = get_eye_center(landmarks, LEFT_EYE_INDICES)
    right_eye_center = get_eye_center(landmarks, RIGHT_EYE_INDICES)
    eyes_center = np.mean([left_eye_center, right_eye_center], axis=0)
    
    nose_bridge = np.array([landmarks.part(30).x, landmarks.part(30).y])
    face_width = landmarks.part(16).x - landmarks.part(0).x
    
    distance = np.linalg.norm(eyes_center - nose_bridge)
    looking_at_camera = (distance / face_width) < GAZE_THRESHOLD
    
    return looking_at_camera, "Looking at camera" if looking_at_camera else "Not looking at camera"

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
        gaze_status = "No face detected"
        for face in faces:
            landmarks = predictor(gray, face)
            looking_at_camera, gaze_status = detect_gaze(landmarks)
            gaze_history.append(looking_at_camera)
            
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Smooth the gaze detection using a moving average
        looking_at_camera = sum(gaze_history) > len(gaze_history) // 2
        
        cv2.putText(frame, gaze_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()