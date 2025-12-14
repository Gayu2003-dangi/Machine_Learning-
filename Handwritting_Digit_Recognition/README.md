pip install numpy==1.23.5




Package	Version
Python	3.10
TensorFlow	2.10.1
NumPy	1.23.5


# 🧠 Handwritten Digit Recognition using TensorFlow

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/TensorFlow-2.10.1-orange" />
  <img src="https://img.shields.io/badge/NumPy-1.23.5-green" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

A **GitHub-ready Machine Learning project** that recognizes handwritten digits (0–9) using a **Deep Neural Network** trained on the **MNIST dataset** with **TensorFlow and Keras**.

---


---

## 🔍 Overview

Handwritten Digit Recognition is a **classic deep learning problem**. This project demonstrates how to:

* Load and preprocess image data
* Build and train a neural network
* Evaluate model performance
* Achieve **~99% accuracy** on unseen test data

This project is ideal for **students, beginners, and portfolio building**.

---

## 🎥 Demo Output

```
Epoch 5/5
loss: 0.0177 - accuracy: 0.9944
val_accuracy: 0.9900

Test Accuracy: 0.99
```

---

## 🛠 Tech Stack

| Category            | Technology        |
| ------------------- | ----------------- |
| Language            | Python 3.10       |
| ML Framework        | TensorFlow 2.10.1 |
| Numerical Computing | NumPy 1.23.5      |
| IDE                 | VS Code           |
| OS                  | Windows           |


pip install numpy==1.23.5




Package	Version
Python	3.10
TensorFlow	2.10.1
NumPy	1.23.5
---

## 📊 Dataset

**MNIST Dataset**

* 70,000 grayscale images of handwritten digits
* 28 × 28 pixel resolution
* Automatically downloaded via TensorFlow

| Split    | Images |
| -------- | ------ |
| Training | 60,000 |
| Testing  | 10,000 |

---

## 🧠 Model Architecture

* Flatten Layer (28×28 → 784)
* Dense Layer with ReLU activation
* Output Layer with Softmax activation (10 classes)


```






### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install tensorflow-cpu==2.10.1
pip install numpy==1.23.5
```

---

## ▶️ Usage

Run the main script:

```bash
python digit_recognition.py
```

T

---

## ✅ Results

* **Training Accuracy:** ~99.4%
* **Test Accuracy:** ~99%
* Stable loss and high generalization

---

## 📁 Project Structure

```
Handwritting_Digit_Recognition/
│
├── digit_recognition.py
├── README.md
└── venv/ (optional)
```

---

## 🚀 Future Enhancements

* Predict digits from custom images
* Add GUI using Tkinter or Streamlit
* Save and load trained models
* Deploy as a web application

---

## 👩‍💻 Author

**Gayatri Dangi**
📍 India

---

## 📜 License

This project is licensed for **educational and learning purposes**.

---

⭐ If you found this project helpful, don’t forget to **star the repository**!
