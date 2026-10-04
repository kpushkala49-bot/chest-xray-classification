# Chest X-ray Classification

## Project Overview

This project uses deep learning and transfer learning techniques to classify chest X-ray images into three classes:

- Lung Opacity
- No Lung Opacity / Not Normal
- Normal

## Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow
- Scikit-learn
- Docker

## Model

The project compares a CNN developed from scratch with pretrained CNN architectures using transfer learning.

The selected final model is saved as:

best_xray_model.keras

## Deployment

The application is developed using Streamlit and packaged using Docker.

Users can upload a chest X-ray image and receive:

- Predicted class
- Prediction probability
- Class-wise probabilities

## Disclaimer

This application is intended for educational and project demonstration purposes only. It is not a medical diagnostic tool.
