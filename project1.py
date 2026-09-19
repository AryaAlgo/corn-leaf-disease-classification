import os
import cv2
import numpy as np

name = "Aryan Kumar"

# Create output folders
os.makedirs("HEALTHY", exist_ok=True)
os.makedirs("UNHEALTHY", exist_ok=True)

def load_images(folder):
    images = []
    names = []
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(folder, file))
            if img is not None:
                images.append(img)
                names.append(file)
    return images, names

# Load dataset
images, names = load_images("dataset")

# Process images
for img, img_name in zip(images, names):

    h, w, _ = img.shape

    blur = cv2.GaussianBlur(img, (3, 3), 2)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)

    lower = np.array([0, 125, 0])
    upper = np.array([255, 255, 255])

    mask = cv2.inRange(lab, lower, upper)

    infected_pixels = np.sum(mask == 255)
    total_pixels = h * w

    percent = (infected_pixels / total_pixels) * 100

    print(f"{img_name}  Infection: {percent:.2f}%")

    if percent >= 40:
        print("Unhealthy")
        cv2.imwrite("UNHEALTHY/" + img_name, img)
    else:
        print("Healthy")
        cv2.imwrite("HEALTHY/" + img_name, img)

print("\nAll images processed successfully!")
