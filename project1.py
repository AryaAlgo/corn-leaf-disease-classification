import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# Corn Leaf Disease Classification using KNN
#
# Dataset:
#   dataset/
#   ├── healthy images/
#   └── unhealthy images/
#
# Label:
#   0 = Healthy
#   1 = Unhealthy
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HEALTHY_DIR = os.path.join(BASE_DIR, "dataset", "healthy images")
UNHEALTHY_DIR = os.path.join(BASE_DIR, "dataset", "unhealthy images")


def extract_features(image):
    """
    Extract image-level color features.

    For each RGB, HSV and LAB channel, calculate:
    - mean
    - standard deviation

    This gives one feature vector per image.
    """
    image = cv2.resize(image, (100, 100))

    # OpenCV reads images as BGR
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    features = []

    # RGB features
    for channel in cv2.split(rgb):
        features.extend([np.mean(channel), np.std(channel)])

    # HSV features
    for channel in cv2.split(hsv):
        features.extend([np.mean(channel), np.std(channel)])

    # LAB features
    for channel in cv2.split(lab):
        features.extend([np.mean(channel), np.std(channel)])

    return features


def load_images_from_folder(folder_path, label):
    """Load images and return one feature vector per image."""
    data = []

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(folder_path, filename)

            image = cv2.imread(image_path)

            if image is None:
                print(f"Skipping unreadable image: {filename}")
                continue

            features = extract_features(image)

            data.append({
                "filename": filename,
                "label": label,
                "features": features
            })

    return data


# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------

print("Loading healthy images...")
healthy_data = load_images_from_folder(HEALTHY_DIR, 0)

print("Loading unhealthy images...")
unhealthy_data = load_images_from_folder(UNHEALTHY_DIR, 1)

if not healthy_data:
    raise ValueError("No readable images found in 'healthy images'.")

if not unhealthy_data:
    raise ValueError("No readable images found in 'unhealthy images'.")


# Combine both classes
all_data = healthy_data + unhealthy_data

X = np.array([item["features"] for item in all_data])
y = np.array([item["label"] for item in all_data])

filenames = np.array([item["filename"] for item in all_data])

print(f"\nTotal images: {len(all_data)}")
print(f"Healthy images: {len(healthy_data)}")
print(f"Unhealthy images: {len(unhealthy_data)}")
print(f"Feature matrix shape: {X.shape}")


# ------------------------------------------------------------
# Split IMAGES into training and testing sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X,
    y,
    filenames,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# Train KNN classifier
# ------------------------------------------------------------

k = 7

knn_classifier = KNeighborsClassifier(n_neighbors=k)

print("\nTraining KNN classifier...")
knn_classifier.fit(X_train, y_train)


# ------------------------------------------------------------
# Make predictions
# ------------------------------------------------------------

y_pred = knn_classifier.predict(X_test)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("KNN Classification Results")
print("==============================")

print(f"Training images: {len(X_train)}")
print(f"Testing images:  {len(X_test)}")
print(f"K value: {k}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Healthy", "Unhealthy"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# Show individual test predictions
# ------------------------------------------------------------

print("\nTest Image Predictions:")

for filename, actual, predicted in zip(files_test, y_test, y_pred):

    actual_label = "Healthy" if actual == 0 else "Unhealthy"
    predicted_label = "Healthy" if predicted == 0 else "Unhealthy"

    print(f"{filename} -> Actual: {actual_label}, Predicted: {predicted_label}")

print("\nAll images processed successfully!")
