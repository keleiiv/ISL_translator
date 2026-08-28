"""Record labelled hand-landmark sequences for training."""

from __future__ import annotations

import argparse

import cv2
import mediapipe as mp
import numpy as np

from config import KEYPOINTS_DIR, PHRASE_VOCABULARY, SEQUENCE_LENGTH
from landmarks import extract_hand_landmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sequences for one ISL gesture.")
    parser.add_argument("label", choices=PHRASE_VOCABULARY, help="Complete banking or hospital phrase to record")
    parser.add_argument("--sequences", type=int, default=20, help="Number of samples to record")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    label = args.label

    start_index = len([p for p in (KEYPOINTS_DIR / label).glob("*") if p.is_dir()])
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the selected webcam.")

    hands_api = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    with hands_api.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
        for sequence_offset in range(args.sequences):
            sequence_index = start_index + sequence_offset
            sequence_dir = KEYPOINTS_DIR / label / str(sequence_index)
            sequence_dir.mkdir(parents=True, exist_ok=True)
            print(f"Press SPACE to record {label}, sample {sequence_offset + 1}/{args.sequences}. Press q to quit.")

            while True:
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    raise SystemExit("Could not read from webcam.")
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "SPACE: record | q: quit", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("ISL data collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return
                if key == ord(" "):
                    break

            for frame_index in range(SEQUENCE_LENGTH):
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.multi_hand_landmarks:
                    drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], hands_api.HAND_CONNECTIONS)
                np.save(sequence_dir / f"{frame_index}.npy", extract_hand_landmarks(result))
                cv2.putText(frame, f"{label}: {frame_index + 1}/{SEQUENCE_LENGTH}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("ISL data collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return

    cap.release()
    cv2.destroyAllWindows()
    print("Collection complete.")


if __name__ == "__main__":
    main()
