# Face Mask Detection System 😷

## 📌 Project Overview
The **Face Mask Detection System** is a computer vision project that detects whether a person is wearing a face mask or not using a webcam or image input.  
This project uses **TensorFlow**, **Keras**, and **OpenCV** to build and deploy a deep learning–based solution.

---

## 🎯 Objective
To develop an automated system that can identify people wearing masks and those not wearing masks in real time for public safety and health monitoring.

---

## 🛠️ Technologies Used
- Python  
- TensorFlow & Keras  
- OpenCV  
- NumPy  
- Scikit-learn  
- VS Code  

---

## 📂 Project Structure
Face_Mask_Detection/
│
├── dataset/
│ ├── with_mask/
│ └── without_mask/
│
├── train_mask_detector.py
├── detect_mask.py
├── mask_detector.model
├── requirements.txt
└── README.md


---

## 📊 Dataset
The dataset contains images of faces in two categories:
- **With Mask**
- **Without Mask**

Dataset source:
- Kaggle Face Mask Dataset  
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

---

## ⚙️ Installation & Setup

### 1️⃣ Clone or Download the Project
```bash
git clone <repository-url>
cd Face_Mask_Detection
🚀 How to Run the Project
🔹 Step 1: Train the Model



python train_mask_detector.py


This will train a CNN model and save it as mask_detector.model.

🔹 Step 2: Run Real-Time Face Mask Detection
python detect_mask.py


Webcam will start automatically

Press Q to exit


🖥️ Output

🟢 Green Box → Mask Detected

🔴 Red Box → No Mask Detected

📚 Skills Gained

Image preprocessing and augmentation

Convolutional Neural Networks (CNN)

Real-time object detection

Model training and evaluation

Computer vision with OpenCV