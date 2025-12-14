import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# -------------------------------
# Load images and labels
# -------------------------------

data = []
labels = []

BASE_DIR = r"C:\Users\hp\Documents\face_mask_detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

for category in CATEGORIES:
    path = os.path.join(BASE_DIR, category)
    label = 0 if category == "with_mask" else 1

    if not os.path.exists(path):
        print(f"❌ Folder not found: {path}")
        continue

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.resize(image, (128, 128))
        data.append(image)
        labels.append(label)

# -------------------------------
# Convert to NumPy arrays
# -------------------------------

data = np.array(data, dtype="float32") / 255.0
labels = to_categorical(labels, 2)

print("Images loaded:", data.shape)
print("Labels shape:", labels.shape)

# -------------------------------
# Build CNN model
# -------------------------------

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(2, activation="softmax")
])

# -------------------------------
# Compile & Train
# -------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    data,
    labels,
    epochs=10,
    batch_size=32
)

# -------------------------------
# Save model
# -------------------------------

model.save("mask_model.h5")
print("✅ Model saved as mask_model.h5")
