"""Central configuration for the ISL recognition pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_PHRASE_DIR = ROOT_DIR / "dataset" / "common_phrases"
KEYPOINTS_DIR = ROOT_DIR / "keypoints"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "phrase_model.keras"
LABELS_PATH = MODELS_DIR / "phrase_labels.json"
STATIC_MODEL_PATH = MODELS_DIR / "static_phrases_model.joblib"
STATIC_LABELS_PATH = MODELS_DIR / "static_phrase_labels.json"

# These values must remain the same for collection, training, and inference.
SEQUENCE_LENGTH = 30
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
FEATURE_SIZE = LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK

# Each label is one complete, useful message - never an alphabet letter.
# Add a new phrase here before collecting it with collect_gestures.py.
PHRASE_VOCABULARY = (
    "BANK_BALANCE",
    "CASH_WITHDRAWAL",
    "MONEY_TRANSFER",
    "HOSPITAL_DOCTOR",
    "MEDICINE",
    "EMERGENCY",
)

# Static service-counter signs available in dataset/common_phrases.
STATIC_PHRASE_LABELS = (
    "hello", "namaste", "please", "yes", "no", "sorry", "thanks",
    "understand", "water", "food",
)
