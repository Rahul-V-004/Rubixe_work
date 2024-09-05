import cv2
import dlib
import numpy as np

# Load dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_gaze(landmarks):
    # Get the left and right eye coordinates from landmarks
    left_eye_indices = [36, 37, 38, 39, 40, 41]
    right_eye_indices = [42, 43, 44, 45, 46, 47]

    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_indices])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_indices])

    # Calculate the center of each eye
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)

    # Calculate the midpoint between the two eye centers
    eyes_center = np.mean([left_eye_center, right_eye_center], axis=0)

    # Consider looking at the camera if the eye center is close to the nose bridge (landmark 30)
    nose_bridge = np.array([landmarks.part(30).x, landmarks.part(30).y])

    # Distance between the nose bridge and eyes center
    distance = np.linalg.norm(eyes_center - nose_bridge)

    # Normalize the distance by the face width
    face_width = np.linalg.norm([landmarks.part(16).x - landmarks.part(0).x])
    gaze_threshold = 0.25  # This can be adjusted

    if distance / face_width < gaze_threshold:
        return True
    return False

def main():
    cap = cv2.VideoCapture(0)  # Use the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        looking_at_camera = False
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)

            # Check if the person is looking at the camera
            looking_at_camera = detect_gaze(landmarks)

            # Draw a rectangle around the face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display result
        text = "Looking at camera" if looking_at_camera else "Not looking at camera"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
