
# ⛑️ Global Solution Project: PPE Detector (CNN)

**Student:** Giulia Bugatti Fonseca — RM562675
**Class:** 1TIAPF

Project developed for FIAP’s **Global Solution**, under the theme **“Intelligent Technologies for the Future of Work.”**

---

## 1. 🎯 Project Objective

The goal of this project was to develop a **Computer Vision solution** to improve workplace safety, focusing on the **binary classification of helmet usage**.

---

## 2. 🛠️ Technologies Used

* **Python 3.10+**
* **TensorFlow (Keras):** CNN built from scratch
* **Streamlit:** Demonstration web application (`app.py`)

---

## 3. 🗂️ Dataset & Preprocessing: Critical Data Engineering

**Source:** Hard Hat Workers Object Detection Dataset (Roboflow)

### Challenges Addressed (Data Engineering)

* **Noise Filtering (Label Ambiguity):**
  Removal of **531 samples** with contradictory labels (`head = 1` and `helmet = 1`).

* **Class Balancing (Undersampling):**
  The original dataset had a **10:1 imbalance** (4301 *helmet* vs. 437 *head*).
  Undersampling was applied to balance the training/validation set to **437 samples per class**, resulting in a total of **874 images**.

---

## 4. 🏗️ CNN Architecture (Built from Scratch)

The final model is a simplified and stabilized **binary classification architecture** (sigmoid output):

* **Stability Improvements:**

  * Added **BatchNormalization** layers
  * Replaced `Flatten` with **GlobalAveragePooling2D** to prevent gradient explosion and numerical instability observed in early experiments

* **Result:**
  The trained model uses only **24,001 parameters**, making it extremely **lightweight and efficient**.

---

## 5. 📊 Results & Analysis

The best performance was achieved after **40 training epochs**.

* **Training Accuracy:** 83.19%
* **Final Validation Accuracy:** 84.57%

### Learning Curves & Critical Limitations

Analysis of the learning curves shows that, despite the robust overall performance of **84.57%**, the model exhibits **generalization limitations** for the **“No Helmet” (head)** class.

Experiments revealed that **Data Augmentation (oversampling techniques)** and extended training were unable to overcome the **low diversity and critically small number of only 437 unique samples** in the *No Helmet* class.
The model has effectively reached a **data-imposed performance ceiling**.

---

## 6. ⚠️ Next Steps (Market-Ready Solution)

* **Data Acquisition:**
  Significantly expand the dataset for the *No Helmet* class (minimum **3,000 new unique samples**).

* **Detection Model:**
  Apply **Transfer Learning** with pre-trained architectures (e.g., **MobileNetV2**) to build an **Object Detection model** (YOLO / SSD) capable of **locating workers in images**, not just classifying cropped inputs.

---

## 7. 🚀 Demonstration (Streamlit)

The trained model (`meu_modelo_epi.keras`) was deployed in a simple **Streamlit web application**, including a fix for **label inversion in binary_crossentropy**, ensuring an intuitive and user-friendly result display.

---

## 8. ⚙️ How to Run the Project

* Keep all files (notebook, `app.py`, `requirements.txt`, model) in the same directory.
* Create and activate a virtual environment:

  ```bash
  python -m venv venv
  ```
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

### Run the web application

```bash
streamlit run app.py
```

### Retrain the model

Open and execute `treinamento_cnn.ipynb`
(**Google Colab recommended**)

---
