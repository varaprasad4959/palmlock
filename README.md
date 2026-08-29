# PalmLock 🔐

PalmLock is a webcam-based biometric authentication system that uses palm recognition to verify a user's identity.

## 🚧 Project Status
Currently under development.

## 🎯 Goal

PalmLock aims to provide passwordless authentication using a user's palm as a biometric identifier.

## 🧠 Planned Architecture

Camera
↓
Hand Detection
↓
Palm Extraction
↓
Palm Recognition
↓
Liveness Detection
↓
Authentication
↓
Unlock

## 🛠️ Technologies

- Python
- PyTorch
- CUDA
- YOLO
- OpenCV
- Computer Vision

## 🔐 Privacy

Palm biometric data is intended to remain locally on the user's device.

Personal enrollment images and biometric templates are excluded from version control.

## 📁 Project Structure

```text
PalmLock/
├── data/
│   └── enrollment/
├── models/
├── src/
├── .gitignore
├── README.md
└── requirements.txt