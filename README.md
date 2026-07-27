# Fresh-vs-Rotten-mango-classifier-FE8

# 🥭 Mango Freshness Classification Using Deep Learning

A Streamlit web application that uses a Convolutional Neural Network (CNN) to classify mangoes as **Fresh** or **Rotten** from an uploaded image.

---

##  Overview

This project applies deep learning to automate mango freshness detection. Users can upload an image of a mango through a simple web interface, and the trained CNN model predicts whether the fruit is fresh or rotten while displaying the prediction confidence.

The application is built with **TensorFlow/Keras** for inference and **Streamlit** for the user interface.

---

## Features

* Upload mango images (`.jpg`, `.jpeg`, `.png`)
* Classify mangoes as **Fresh** or **Rotten**
* Display prediction confidence
* Show class probabilities
* Modern and responsive Streamlit interface
* Fast inference using a pre-trained Keras model
* Cached model loading for improved performance

---

## Technologies Used

* Python 3.x
* TensorFlow / Keras
* Streamlit
* NumPy
* Pillow (PIL)

---

## Project Structure

```
project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── custom_cnn_best(1).keras
│
└── result/
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/mango-freshness-classifier.git

cd mango-freshness-classifier
```

### 2. Create a virtual environment (Optional)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit tensorflow pillow numpy
```

---

## Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

After running the command, Streamlit will open the application in your default web browser.

---

## Model

The application loads the trained CNN model from:

```
models/custom_cnn_best(1).keras
```

The model was trained to classify mango images into two categories:

* Fresh Mango
* Rotten Mango

---

## How to Use

1. Launch the Streamlit application.
2. Upload a mango image.
3. Click **Classify Mango**.
4. View:

   * Predicted class
   * Prediction confidence
   * Probability for each class

---

## Example Output

```
Prediction:
Fresh Mango

Confidence:
96.42%

Fresh Probability:
96.42%

Rotten Probability:
3.58%
```

---