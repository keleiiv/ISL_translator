"""Train a small static service-counter phrase classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from config import MODELS_DIR, STATIC_LABELS_PATH, STATIC_MODEL_PATH, STATIC_PHRASE_DIR, STATIC_PHRASE_LABELS

HAND_FEATURE_SIZE = 126  # left and right MediaPipe hands, 21 landmarks x 3 each


def load_static_phrases(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Use five evenly spaced static frames from every labelled sample folder."""
    features: list[np.ndarray] = []
    targets: list[int] = []
    missing = [label for label in STATIC_PHRASE_LABELS if not (data_dir / label).is_dir()]
    if missing:
        raise SystemExit(f"Missing required phrase folders: {', '.join(missing)}")

    for target, label in enumerate(STATIC_PHRASE_LABELS):
        for sample_dir in sorted(path for path in (data_dir / label).iterdir() if path.is_dir()):
            frames = sorted(sample_dir.glob("*.npy"), key=lambda path: int(path.stem))
            for frame_path in np.asarray(frames, dtype=object)[np.linspace(0, len(frames) - 1, 5, dtype=int)]:
                vector = np.load(frame_path).astype(np.float32).reshape(-1)
                if vector.size != 1662:
                    raise SystemExit(f"Unexpected landmark shape in {frame_path}: {vector.size}")
                features.append(vector[-HAND_FEATURE_SIZE:])
                targets.append(target)

    return np.stack(features), np.asarray(targets), list(STATIC_PHRASE_LABELS)


def main() -> None:
    features, targets, labels = load_static_phrases(STATIC_PHRASE_DIR)
    model = KNeighborsClassifier(n_neighbors=3, weights="distance")
    model.fit(features, targets)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, STATIC_MODEL_PATH)
    STATIC_LABELS_PATH.write_text(json.dumps(labels), encoding="utf-8")
    print(f"Trained {len(labels)} phrases from {len(features)} hand-landmark samples.")
    print(f"Saved model to {STATIC_MODEL_PATH}")


if __name__ == "__main__":
    main()
