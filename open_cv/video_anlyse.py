import cv2
import dlib
import numpy as np
import json
from scipy.spatial import distance as dist

# Load the face detection and landmark prediction models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("open_cv/shape_predictor_68_face_landmarks.dat")

# Constants for gaze detection
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
GAZE_THRESHOLD = 0.25
EYE_AR_THRESH = 0.2  # Eye Aspect Ratio threshold

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Gaze detection function
def detect_gaze(landmarks):
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES])
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    
    if avg_ear < EYE_AR_THRESH:
        return False, "Eyes Closed"
    
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    eyes_center = np.mean([left_eye_center, right_eye_center], axis=0)
    
    nose_bridge = np.array([landmarks.part(30).x, landmarks.part(30).y])
    face_width = landmarks.part(16).x - landmarks.part(0).x
    
    distance = np.linalg.norm(eyes_center - nose_bridge)
    looking_at_camera = (distance / face_width) < GAZE_THRESHOLD
    
    return looking_at_camera, "Looking at camera" if looking_at_camera else "Not looking at camera"

# Load the video
video_filename = 'recorded_video.avi'
cap = cv2.VideoCapture(video_filename)

# Variables to track the gaze status
total_frames = 0
looking_at_camera_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        looking_at_camera, _ = detect_gaze(landmarks)
        if looking_at_camera:
            looking_at_camera_frames += 1
        total_frames += 1

# Calculate the percentage of frames where the person is looking at the camera
percentage_looking_at_camera = (looking_at_camera_frames / total_frames) * 100 if total_frames else 0

# Prepare the report data as a dictionary
report_data = {
    "total_frames": total_frames,
    "frames_looking_at_camera": looking_at_camera_frames,
    "percentage_looking_at_camera": round(percentage_looking_at_camera, 2),
    "conclusion": "Success" if percentage_looking_at_camera >= 80 else "Failed"
}

# Write the report to a JSON file
with open("gaze_report.json", "w") as json_file:
    json.dump(report_data, json_file, indent=4)

# Release resources
cap.release()
