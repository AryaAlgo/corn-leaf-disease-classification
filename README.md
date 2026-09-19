# Corn Leaf Disease Classification

A machine learning-based image classification project for identifying **healthy and unhealthy corn leaves** using image processing and the **K-Nearest Neighbors (KNN)** algorithm.

## Objective

The objective of this project is to develop a simple machine learning pipeline for classifying corn leaf images into two categories:

- **Healthy**
- **Unhealthy**

The project uses image processing to extract color-based features from corn leaf images and applies the **K-Nearest Neighbors (KNN)** algorithm for classification.

## Dataset

The dataset consists of **20 corn leaf images**:

- 10 healthy corn leaf images
- 10 unhealthy corn leaf images

The unhealthy images show visible disease symptoms, while the healthy images represent normal corn leaves.

## Methodology

The classification pipeline uses color features extracted from multiple color spaces:

- **RGB** — Red, Green, and Blue
- **HSV** — Hue, Saturation, and Value
- **LAB** — Lightness and color-opponent dimensions

These extracted features are used as input to the KNN machine learning model.

## Workflow

The overall workflow of the project is:

1. Collect corn leaf images.
2. Classify images into healthy and unhealthy categories.
3. Extract color-based features from RGB, HSV, and LAB color spaces.
4. Store the extracted features in a Pandas DataFrame.
5. Split the data into training and testing sets.
6. Train the KNN classification model.
7. Use the trained model to classify corn leaf images.

## Technologies Used

- **Python** — Programming language
- **OpenCV** — Image processing and color feature extraction
- **Pandas** — Data organization and feature handling
- **NumPy** — Numerical operations
- **Scikit-learn** — KNN machine learning model

## Machine Learning Model

The project uses the **K-Nearest Neighbors (KNN)** algorithm for classification.

The extracted RGB, HSV, and LAB color features are provided as input to the KNN model to classify each image as:

- **Healthy**
- **Unhealthy**

## Results

The KNN model is trained using the extracted RGB, HSV, and LAB color features to classify corn leaf images into healthy and unhealthy categories.

## Project Structure

```text
corn-leaf-disease-classification/
│
├── README.md
├── LICENSE
├── .gitignore
├── project1.py
│
└── dataset/
    ├── healthy images
    └── unhealthy images
