"""Recognize trained static service-counter phrases from a webcam."""

from __future__ import annotations

import json
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp

from config import STATIC_LABELS_PATH, STATIC_MODEL_PATH
from landmarks import extract_two_hand_landmarks


def main() -> None:
    if not STATIC_MODEL_PATH.exists() or not STATIC_LABELS_PATH.exists():
        raise SystemExit("No static phrase model found. Run: python src/train_static_phrases.py")
    model = joblib.load(STATIC_MODEL_PATH)
    labels = json.loads(STATIC_LABELS_PATH.read_text(encoding="utf-8"))
    recent: deque[str] = deque(maxlen=7)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    holistic_api = mp.solutions.holistic
    drawing = mp.solutions.drawing_utils
    with holistic_api.Holistic(model_complexity=0, refine_face_landmarks=False) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
                if hand:
                    drawing.draw_landmarks(frame, hand, holistic_api.HAND_CONNECTIONS)

            if result.left_hand_landmarks or result.right_hand_landmarks:
                predicted = labels[int(model.predict([extract_two_hand_landmarks(result)])[0])]
                recent.append(predicted)
            text = Counter(recent).most_common(1)[0][0].upper() if recent else "Show a sign"
            cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("ISL static phrases - q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
