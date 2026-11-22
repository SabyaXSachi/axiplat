"""Train the CNN age regressor on a real face dataset.

Expects UTKFace-style images (named AGE_GENDER_RACE_DATETIME.jpg, e.g.
"25_0_1_201701161834.jpg") in a dataset/ folder next to this script.
Refuses to train when no dataset is found: a model trained on anything
else (e.g. random pixels) would silently corrupt the age fusion in
app.py, which blends the CNN in at 40% weight when model/cnn_age_model.h5
exists.

Dataset: UTKFace aligned & cropped set
(https://susanqq.github.io/UTKFace/) - download and unzip into dataset/.
"""

import os
import sys

import numpy as np

DATASET_DIR = "dataset"
MODEL_PATH = os.path.join("model", "cnn_age_model.h5")
IMG_SIZE = 64
MAX_IMAGES = 20000
MAX_AGE = 90


def load_dataset():
    import cv2

    images, ages = [], []
    for name in sorted(os.listdir(DATASET_DIR)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        age_part = name.split("_")[0]
        if not age_part.isdigit():
            continue  # UTKFace names start with the age
        age = int(age_part)
        if age > MAX_AGE:
            continue
        img = cv2.imread(os.path.join(DATASET_DIR, name))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        ages.append(age)
        if len(images) >= MAX_IMAGES:
            break
    return images, ages


def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("TensorFlow is not installed. Install it with:")
        print("  pip install tensorflow")
        sys.exit(1)

    if not os.path.isdir(DATASET_DIR):
        print(f"No '{DATASET_DIR}/' folder found.")
        print("Download the UTKFace aligned dataset, unzip it into dataset/, then rerun.")
        print("Refusing to train on fake data: the fusion in app.py would be corrupted.")
        sys.exit(1)

    print("Loading dataset...")
    images, ages = load_dataset()
    if len(images) < 100:
        print(f"Only {len(images)} usable images found in '{DATASET_DIR}/'.")
        print("Need at least 100 UTKFace-style images (AGE_...jpg names).")
        sys.exit(1)

    X = np.asarray(images, dtype="float32") / 255.0
    y = np.asarray(ages, dtype="float32")
    print(f"Loaded {len(X)} images (age range {int(y.min())}-{int(y.max())})")

    model = build_model()

    # Hold out a validation split so the reported MAE is honest
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
    )

    val_mae = model.evaluate(X_val, y_val, verbose=0)[1]
    print(f"\nValidation MAE: {val_mae:.2f} years")

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    main()
