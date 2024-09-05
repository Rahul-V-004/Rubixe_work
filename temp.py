import cv2
import numpy as np
import os

FILE_OUTPUT = 'output.mp4'

# Checks and deletes the output file
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Capturing video from webcam:
cap = cv2.VideoCapture(0)
currentFrame = 0

# Get current width and height of the frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID is more common; you can also use 'X264' if you have the codec installed
out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 20.0, (width, height))

# Start capturing video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Handle the mirroring of the current frame
        frame = cv2.flip(frame, 1)

        # Write the flipped frame to the output video
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
