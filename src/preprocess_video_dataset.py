"""Convert ISL videos into normalized two-hand MediaPipe landmark sequences."""

from __future__ import annotations

from pathlib import Path
import warnings

import cv2
import mediapipe as mp
import numpy as np

from config import KEYPOINTS_DIR, ROOT_DIR, SEQUENCE_LENGTH


VIDEO_CLASSES = ("eat", "go", "hello", "help", "no", "please", "water", "yes")
VIDEO_DIR = ROOT_DIR / "dataset" / "isl_video"
OUTPUT_DIR = KEYPOINTS_DIR / "isl_video"
HAND_SHAPE = (21, 3)


def normalize_hand(hand_landmarks) -> np.ndarray:
    """Center a hand at its wrist and scale it by its wrist-to-middle-MCP span."""
    points = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
        dtype=np.float32,
    )
    centered = points - points[0]
    scale = np.linalg.norm(centered[9])
    if scale < 1e-6:
        scale = np.linalg.norm(centered, axis=1).max()
    return centered / scale if scale >= 1e-6 else np.zeros(HAND_SHAPE, dtype=np.float32)


def extract_frame_landmarks(results) -> np.ndarray:
    """Return normalized left/right hands; missing hands remain zero-padded."""
    frame = np.zeros((2, *HAND_SHAPE), dtype=np.float32)
    if not results.multi_hand_landmarks:
        return frame

    handedness = results.multi_handedness or []
    for landmarks, handed in zip(results.multi_hand_landmarks, handedness):
        label = handed.classification[0].label
        slot = 0 if label == "Left" else 1
        frame[slot] = normalize_hand(landmarks)
    return frame


def process_video(video_path: Path, hands) -> np.ndarray | None:
    """Sample and process exactly ``SEQUENCE_LENGTH`` evenly spaced frames."""
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or frame_count < 1:
        capture.release()
        warnings.warn(f"Skipping unreadable video: {video_path}")
        return None

    sampled_indices = np.linspace(0, frame_count - 1, SEQUENCE_LENGTH, dtype=int)
    sequence = []
    for frame_index in sampled_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, image = capture.read()
        if not ok:
            capture.release()
            warnings.warn(f"Skipping unreadable sampled frame in: {video_path}")
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sequence.append(extract_frame_landmarks(hands.process(rgb)))

    capture.release()
    return np.stack(sequence)


def main() -> None:
    total_processed = 0
    total_skipped = 0
    hands_api = mp.solutions.hands
    with hands_api.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7,
    ) as hands:
        for label in VIDEO_CLASSES:
            class_dir = VIDEO_DIR / label
            videos = sorted(class_dir.rglob("*.mp4")) if class_dir.is_dir() else []
            processed = 0
            skipped = 0
            if not videos:
                warnings.warn(f"No MP4 videos found for class: {label}")
            for video_path in videos:
                sequence = process_video(video_path, hands)
                if sequence is None:
                    skipped += 1
                    continue
                output_path = OUTPUT_DIR / label / f"{video_path.stem}.npy"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, sequence)
                processed += 1
            total_processed += processed
            total_skipped += skipped
            print(f"{label}: {processed} processed, {skipped} skipped")

    print(f"Total: {total_processed} processed, {total_skipped} skipped")


if __name__ == "__main__":
    main()
