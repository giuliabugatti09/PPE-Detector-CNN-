# ⛑️ Workplace Safety: Efficient CNN for PPE Detection

> **Edge AI Solution** developed for workplace safety, focusing on real-time binary classification of helmet usage. 

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 1. 🎯 Project Overview
This project addresses industrial safety by implementing a **Computer Vision pipeline** to detect Personal Protective Equipment (PPE). The core objective was to build a highly efficient, lightweight model capable of distinguishing between workers wearing helmets and those without.

---

## 2. 🏗️ Architecture: Lightweight CNN
The model was architected from scratch with a focus on **efficiency and deployment stability**.

* **Global Average Pooling 2D:** Used instead of Flatten to drastically reduce parameters and mitigate overfitting.
* **Batch Normalization:** Integrated for faster convergence and gradient stability.
* **Parameter Count:** Only **24,001 parameters**, making it ideal for **Edge Computing** and low-latency environments.

---

## 3. 🗂️ Data Engineering & Preprocessing
Significant effort was invested in **Data Quality** (Garbage In, Garbage Out):

* **Noise Reduction:** Sanitized the *Hard Hat Workers Dataset* by removing **531 samples** with ambiguous or contradictory labels.
* **Class Balancing:** Addressed a **10:1 imbalance** through strategic **Undersampling**, resulting in a balanced training set of 874 high-quality images.

---

## 4. 📊 Performance Metrics
The model achieved robust results given the data constraints:

* **Training Accuracy:** 83.19%
* **Validation Accuracy:** **84.57%**

> **Analysis of Data Ceiling:** The project identified a performance bottleneck caused by the limited diversity in the "No Helmet" class (437 samples). This transparency in reporting **model limitations** is a key part of the development lifecycle.



---

## 🚀 Future Roadmap (Scalability)
To transform this PoC (Proof of Concept) into a production-ready system:
1.  **Object Detection (YOLO/SSD):** Pivot from classification to real-time detection for multiple workers in a single frame.
2.  **Transfer Learning:** Implement **MobileNetV2** as a backbone to improve feature extraction while maintaining low latency.
3.  **Data Expansion:** Integrate ~3,000 new unique samples for the "No Helmet" class to improve generalization.

---

## ⚙️ Setup & Execution
1. **Environment:** `python -m venv venv`
2. **Install:** `pip install -r requirements.txt`
3. **Run Web App:** `streamlit run app.py`

---
**Giulia Bugatti** 
