"""Load collected ISL landmark sequences safely and consistently."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import FEATURE_SIZE, SEQUENCE_LENGTH


def available_labels(data_dir: Path, allowed_labels: tuple[str, ...] | None = None) -> list[str]:
    """Return gesture labels in a stable order."""
    if not data_dir.exists():
        return []
    found = {entry.name for entry in data_dir.iterdir() if entry.is_dir()}
    return [label for label in allowed_labels if label in found] if allowed_labels else sorted(found)


def load_dataset(
    data_dir: Path, allowed_labels: tuple[str, ...] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load only complete sequences and return features, targets, and labels.

    A sequence is complete only if it has frame files 0.npy through 29.npy.
    This protects training from interrupted webcam captures.
    """
    labels = available_labels(data_dir, allowed_labels)
    sequences: list[np.ndarray] = []
    targets: list[int] = []

    for label_index, label in enumerate(labels):
        label_dir = data_dir / label
        for sequence_dir in sorted((p for p in label_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            frames: list[np.ndarray] = []
            try:
                for frame_index in range(SEQUENCE_LENGTH):
                    frame = np.load(sequence_dir / f"{frame_index}.npy").astype(np.float32)
                    frames.append(frame.reshape(FEATURE_SIZE))
            except (FileNotFoundError, ValueError):
                continue
            sequences.append(np.stack(frames))
            targets.append(label_index)

    if not sequences:
        return (
            np.empty((0, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            labels,
        )
    return np.stack(sequences), np.asarray(targets, dtype=np.int64), labels
