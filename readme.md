# 🚦 Traffic Sign Recognition using CNN

## 📌 Project Overview
This project is a deep learning-based Traffic Sign Recognition system built using Convolutional Neural Networks (CNN).  
It classifies traffic sign images into multiple categories using a trained TensorFlow/Keras model.

---

## 🧠 Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## 📂 Dataset
Used a Traffic Sign Dataset (GTSRB-like structure)
DATA/
├── 0/
├── 1/
├── 2/
├── ...
├── N/

Each folder contains images corresponding to a specific traffic sign category.
---

## 🏗️ Model Architecture
- Convolutional Neural Network (CNN)
- Data Augmentation (rotation, flip, zoom)
- Multiple Conv2D + MaxPooling layers
- Dense fully connected layers
- Softmax output layer

---

## ⚙️ How to Run

### 1. Install dependencies
   pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn

 2. Run training
    python train.py

📊 Results
● Achieved ~95% validation accuracy
● Efficient multi-class image classification
● Good generalization on unseen traffic sign images

🚀 Features
● CNN-based image classification
● Data augmentation for better generalization
● Scalable dataset handling
● Ready for real-time prediction extension

👨‍💻 Author
Shubh Kumar Mishra

📌 Note
This project is part of my Machine Learning portfolio and demonstrates end-to-end deep learning workflow for image classification.
