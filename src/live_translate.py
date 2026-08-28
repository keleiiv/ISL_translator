"""Recognize trained ISL gestures from a webcam feed."""

from __future__ import annotations

import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from config import LABELS_PATH, MODEL_PATH, SEQUENCE_LENGTH
from landmarks import extract_hand_landmarks

CONFIDENCE_THRESHOLD = 0.80


def main() -> None:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise SystemExit("No trained model found. Run: python src/train_model.py")
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(MODEL_PATH)
    frames: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    hands_api = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    with hands_api.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], hands_api.HAND_CONNECTIONS)
            frames.append(extract_hand_landmarks(result))

            text = "Show a gesture..."
            if len(frames) == SEQUENCE_LENGTH:
                probabilities = model.predict(np.expand_dims(np.asarray(frames), axis=0), verbose=0)[0]
                best = int(np.argmax(probabilities))
                if probabilities[best] >= CONFIDENCE_THRESHOLD:
                    text = f"{labels[best]} ({probabilities[best]:.0%})"
            cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("ISL live translator - q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
