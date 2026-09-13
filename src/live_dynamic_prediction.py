import json
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from preprocess_video_dataset import extract_frame_landmarks


MODEL_PATH = Path("models/dynamic_8words_expanded.keras")
LABELS_PATH = Path("models/dynamic_8words_expanded_labels.json")

SEQUENCE_LENGTH = 30
BUFFER_LENGTH = 90

model = tf.keras.models.load_model(MODEL_PATH)

with LABELS_PATH.open("r", encoding="utf-8") as f:
    labels = json.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Keep a longer webcam buffer so we can sample frames
# in a way that is closer to the training preprocessing.
frame_buffer = deque(maxlen=BUFFER_LENGTH)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
) as hands:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    prediction = "Waiting..."
    confidence = 0.0

    print("Webcam started.")
    print("Perform one sign at a time.")
    print("Press Q to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Could not read webcam frame.")
            break

        # Mirror the webcam for a natural selfie view.
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks = extract_frame_landmarks(results)
        frame_buffer.append(landmarks)

        if len(frame_buffer) == BUFFER_LENGTH:

            # Sample 30 evenly spaced frames from the 90-frame buffer.
            indices = np.linspace(
                0,
                BUFFER_LENGTH - 1,
                SEQUENCE_LENGTH,
                dtype=int,
            )

            sampled_sequence = np.asarray(
                [frame_buffer[i] for i in indices],
                dtype=np.float32,
            )

            input_data = sampled_sequence.reshape(1, 30, 126)

            probabilities = model.predict(
                input_data,
                verbose=0,
            )[0]

            index = int(np.argmax(probabilities))

            prediction = labels[index]
            confidence = float(probabilities[index])

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

        cv2.rectangle(
            frame,
            (10, 10),
            (520, 90),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.imshow("ISL Dynamic Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
