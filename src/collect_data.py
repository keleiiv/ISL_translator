import os
import cv2
import numpy as np

from hand_tracking import detect_hands
from extract_landmarks import extract_landmarks

# -------------------------------
# SETTINGS
# -------------------------------
ACTION = "A"          # Change this later (A, B, C...)
NUM_SEQUENCES = 20    # Number of recordings
SEQUENCE_LENGTH = 30  # Frames per recording

DATA_PATH = "keypoints"

# Create folders
for sequence in range(NUM_SEQUENCES):
    os.makedirs(
        os.path.join(DATA_PATH, ACTION, str(sequence)),
        exist_ok=True
    )

cap = cv2.VideoCapture(0)

print("Press ENTER in the terminal before each recording.")
print("Press 'q' in the webcam window to quit.")

for sequence in range(NUM_SEQUENCES):

    input(f"\nReady for Sequence {sequence}? Press ENTER...")

    frame_num = 0

    while frame_num < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            break

        frame, results = detect_hands(frame)

        keypoints = extract_landmarks(results)

        np.save(
            os.path.join(
                DATA_PATH,
                ACTION,
                str(sequence),
                f"{frame_num}.npy"
            ),
            keypoints
        )

        cv2.putText(
            frame,
            f"{ACTION} | Sequence {sequence} | Frame {frame_num}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Collecting Data", frame)

        frame_num += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()

print("Dataset collection complete!")