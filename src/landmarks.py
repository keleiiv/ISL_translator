"""Convert MediaPipe hand results into fixed-size model features."""

from __future__ import annotations

import numpy as np

from config import FEATURE_SIZE


def extract_hand_landmarks(results) -> np.ndarray:
    """Return the first detected hand as a flattened 63-value vector.

    A zero vector represents frames where no hand is visible.  Keeping a fixed
    shape makes every sequence suitable for an LSTM model.
    """
    if not results or not results.multi_hand_landmarks:
        return np.zeros(FEATURE_SIZE, dtype=np.float32)

    hand = results.multi_hand_landmarks[0]
    return np.asarray(
        [[point.x, point.y, point.z] for point in hand.landmark],
        dtype=np.float32,
    ).reshape(FEATURE_SIZE)


def extract_two_hand_landmarks(results) -> np.ndarray:
    """Return left then right hand landmarks as a fixed 126-value vector."""
    features = []
    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand:
            features.extend((point.x, point.y, point.z) for point in hand.landmark)
        else:
            features.extend([(0.0, 0.0, 0.0)] * 21)
    return np.asarray(features, dtype=np.float32).reshape(FEATURE_SIZE * 2)
