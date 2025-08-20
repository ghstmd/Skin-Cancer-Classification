# 🩺 Skin Cancer Classification with Deep Learning

This project focuses on building and evaluating deep learning models for **skin cancer classification** using the [Skin Cancer MNIST (HAM10000) dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

## 🔍 Project Overview
- Performed **exploratory data analysis (EDA)** and preprocessing on image and metadata.  
- Trained and compared **six CNN architectures**:  
  - ResNet50  
  - ResNet18  
  - EfficientNetB0  
  - VGG16  
  - DenseNet121  
  - MobileNetV2  
- Evaluated models under two training setups:  
  1. Using **only images**  
  2. Using **images + metadata**  
- Found **EfficientNetB0** as the best-performing model and fine-tuned it for improved accuracy.  

## 🌐 Web Application
Built an **interactive Flask web app** where users can:
- Upload a skin lesion image (with or without metadata).  
- Get a **predicted disease class** using the trained EfficientNetB0 model.  

## ⚙️ Tech Stack
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)  
- **Deep Learning:** TensorFlow / Keras (and/or PyTorch)  
- **Model Deployment:** Flask  
- **Dataset:** HAM10000 (Skin Cancer MNIST)  

## 🚀 How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/skin-cancer-classification.git
   cd skin-cancer-classification
