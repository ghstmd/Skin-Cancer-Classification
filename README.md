🩺 Skin Cancer Classification with Deep Learning

This project focuses on building and evaluating deep learning models for skin cancer classification using the Skin Cancer MNIST (HAM10000) dataset
.

🔍 Project Overview

Performed exploratory data analysis (EDA) and preprocessing on image and metadata.

Trained and compared six CNN architectures:

ResNet50

ResNet18

EfficientNetB0

VGG16

DenseNet121

MobileNetV2

Evaluated models under two training setups:

Using only images

Using images + metadata

Found EfficientNetB0 as the best-performing model and fine-tuned it for improved accuracy.

🌐 Web Application

Built an interactive Flask web app where users can:

Upload a skin lesion image (with or without metadata).

Get a predicted disease class using the trained EfficientNetB0 model.

⚙️ Tech Stack

Python (Pandas, NumPy, Matplotlib, Seaborn)

Deep Learning: TensorFlow / Keras, PyTorch (if used)

Model Deployment: Flask

Dataset: HAM10000 (Skin Cancer MNIST)

🚀 How to Run

Clone this repository

git clone https://github.com/your-username/skin-cancer-classification.git
cd skin-cancer-classification


Install dependencies

pip install -r requirements.txt


Run the Flask app

python app.py

📊 Results

Best Model: EfficientNetB0 (fine-tuned)

Training with image + metadata provided a performance boost compared to using images alone.
