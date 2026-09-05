from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("keypoints/isl_video")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = ["eat", "go", "hello", "help", "no", "please", "water", "yes"]

X, y = [], []

for class_name in CLASS_NAMES:
    for file in sorted((DATA_DIR / class_name).glob("*.npy")):
        sequence = np.load(file).astype(np.float32)
        if sequence.shape != (30, 2, 21, 3):
            print(f"Skipping unexpected shape: {file} -> {sequence.shape}")
            continue
        X.append(sequence.reshape(30, -1))  # (30, 126)
        y.append(class_name)

X = np.asarray(X, dtype=np.float32)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(f"Loaded sequences: {len(X)}")
print(f"Input shape: {X.shape}")
print("Classes:", list(encoder.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=SEED, stratify=y_encoded
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 126)),
    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )
]

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    callbacks=callbacks,
    verbose=1,
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

model_path = MODEL_DIR / "dynamic_8words.keras"
labels_path = MODEL_DIR / "dynamic_8words_labels.json"

model.save(model_path)
with labels_path.open("w", encoding="utf-8") as f:
    json.dump(list(encoder.classes_), f, indent=2)

print("\nTraining complete.")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Model saved to: {model_path}")
print(f"Labels saved to: {labels_path}")
