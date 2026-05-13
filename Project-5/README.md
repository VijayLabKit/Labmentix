
# 🦅 Aerial Object Intelligence Dashboard

A deep learning-powered web application for **real-time classification of aerial objects (Bird vs Drone)** using Computer Vision and Transfer Learning.

Developed by **Ishan Chowdhury** as a Capstone Project.

---

## 🚀 Project Overview

The Aerial Object Intelligence Dashboard is designed to distinguish between **natural aerial entities (birds)** and **man-made aerial vehicles (drones)** in real-time.

It leverages:
- Transfer Learning (MobileNetV2)
- TensorFlow/Keras
- Streamlit for UI

This system is useful for **airspace monitoring, security systems, and ecological research**.

---

## 🧠 Key Features

- 🖼️ Image classification (Bird vs Drone)
- 📊 Dataset analytics visualization
- 🎯 High accuracy model (~96%)
- 📁 Upload custom images
- 🎲 Random dataset testing
- 📈 Confidence score display
- ⚠️ Alert system for drone detection
- 🌐 Interactive Streamlit dashboard

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- Streamlit
- NumPy, Pandas, PIL

---

## 📂 Project Structure

```
project/
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/
│   ├── transfer_bird_drone.keras
│   └── transfer_bird_drone.h5
│
├── app.py
└── README.md
```

---

## ⚙️ How to Run Locally

### 1. Install dependencies

```
pip install streamlit tensorflow numpy pandas pillow
```

### 2. Run the app

```
streamlit run app.py
```

---

## 📊 Model Details

- Backbone: MobileNetV2
- Input Size: 224x224
- Optimizer: Adam
- Epochs: 15
- Accuracy: ~96.6%

---

## 💼 Use Cases

- ✈️ Airport Safety (bird strike prevention)
- 🛡️ Defense (drone detection)
- 🔬 Research (wildlife monitoring)

---

## 👨‍💻 Author

**Ishan Chowdhury**  
Capstone Project  
Aerial Object Intelligence v1.0

---

## 📌 Notes

- Dataset path must be updated in code
- Model files must be present in `/models` folder
- Supports only image-based prediction (no video yet)

---

## 📄 License

This project is for educational and research purposes.
