from pathlib import Path

import numpy as np

from data_loader import load_dataset


def test_load_dataset_skips_incomplete_sequence(tmp_path: Path):
    complete = tmp_path / "A" / "0"
    complete.mkdir(parents=True)
    for frame in range(30):
        np.save(complete / f"{frame}.npy", np.zeros((21, 3), dtype=np.float32))
    incomplete = tmp_path / "B" / "0"
    incomplete.mkdir(parents=True)
    np.save(incomplete / "0.npy", np.zeros((21, 3), dtype=np.float32))

    features, targets, labels = load_dataset(tmp_path)

    assert labels == ["A", "B"]
    assert features.shape == (1, 30, 63)
    assert targets.tolist() == [0]
