# 🌿 Plant Disease Detection using Machine Learning & Deep Learning

> Comparing KNN, SVM, Random Forest, MLP, and CNN for automated plant disease classification from leaf images.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikit-learn)](https://scikit-learn.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/plant-disease-detection/blob/main/plant_disease_detection_colab.ipynb)
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-green)](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 📌 Project Overview

Every year, crop diseases cause billions of dollars in agricultural losses — especially in regions where farmers don't have quick access to agronomists or plant experts. This project explores whether machine learning can help solve that by automatically identifying plant diseases from a simple photo of a leaf.

I trained and compared **6 different models** (from classical ML to deep learning) on the PlantVillage dataset, applied hyperparameter tuning, and built an interactive web UI so anyone can upload a leaf photo and get an instant diagnosis.

---

## 🎯 What I Built

- Trained **5 classical ML models** — KNN, Decision Tree, Random Forest, SVM, MLP — on PCA-compressed image features
- Built a **4-block CNN** with data augmentation, batch normalisation, dropout regularisation, and callbacks (EarlyStopping, ReduceLROnPlateau)
- Applied **hyperparameter tuning** using GridSearchCV, RandomizedSearchCV, and Keras Tuner
- Created a full **model comparison** across all 8 model variants (before and after tuning)
- Built an **interactive Gradio UI** where users can upload leaf photos and get real-time disease predictions with confidence scores

---

## 📊 Results

| Model | Test Accuracy |
|---|---|
| CNN (Tuned) | **~92%** |
| CNN (Base) | ~89% |
| SVM (Tuned) | ~78% |
| Random Forest (Tuned) | ~74% |
| MLP | ~72% |
| KNN (Tuned) | ~65% |
| KNN (Default) | ~61% |
| Decision Tree | ~52% |

> Exact numbers will vary depending on the random seed and how many images per class are loaded.

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Click the **Open in Colab** badge at the top of this README
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells with **Runtime → Run all**

The notebook will:
- Download the dataset automatically from Kaggle (you'll need to upload your `kaggle.json` API key)
- Train all models
- Show comparison charts
- Launch a Gradio web UI with a public link

### Option 2 — Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
jupyter notebook plant_disease_detection_colab.ipynb
```

---

## 🧪 Dataset

**PlantVillage Dataset** — [Kaggle link](https://www.kaggle.com/datasets/emmarex/plantdisease)

- 38 classes of plant conditions (healthy + diseased)
- 54,000+ leaf images
- Plants covered: Tomato, Potato, Pepper, Apple, Grape, Corn, and more
- I use up to 300 images per class for training (to keep Colab memory manageable)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Classical ML | scikit-learn |
| Hyperparameter Tuning | Keras Tuner, GridSearchCV, RandomizedSearchCV |
| Image Processing | OpenCV, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Web UI | Gradio |
| Environment | Google Colab |

---

## 🧠 Key ML Concepts Covered

- **PCA** (Principal Component Analysis) for dimensionality reduction
- **Data augmentation** to improve CNN generalisation
- **Batch normalisation** and **dropout** for regularisation
- **Early stopping** and **learning rate scheduling**
- **GridSearchCV** and **RandomizedSearchCV** for classical model tuning
- **Neural Architecture Search** with Keras Tuner
- **Confusion matrix** and **classification report** for evaluation

---

## 📸 Screenshots

### Model Comparison
*(Add model_comparison.png here after running the notebook)*

### Gradio UI
*(Add gradio_ui_screenshot.png here)*

---

## 💡 Future Improvements

- Use a pretrained model (ResNet50 / EfficientNet) with transfer learning for higher accuracy
- Deploy the Gradio app permanently on Hugging Face Spaces
- Add a mobile-friendly version using TensorFlow Lite
- Extend to more plant species not covered in PlantVillage



Built as part of my ML learning journey. Dataset credit: PlantVillage via Kaggle.
