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
This project was implemented using the following libraries and frameworks:

- [NumPy](https://numpy.org/) – numerical computing  
- [Pandas](https://pandas.pydata.org/) – data manipulation & analysis  
- [SciPy](https://scipy.org/) – scientific computing  
- [Matplotlib](https://matplotlib.org/) – data visualization  
- [Seaborn](https://seaborn.pydata.org/) – statistical data visualization  
- [Pillow (PIL)](https://pillow.readthedocs.io/) – image processing  
- [ImageHash](https://pypi.org/project/ImageHash/) – perceptual hashing for images  
- [scikit-learn](https://scikit-learn.org/stable/) – ML utilities & metrics  
- [tqdm](https://tqdm.github.io/) – progress bars  
- [Keras](https://keras.io/) – high-level deep learning API  
- [TensorFlow](https://www.tensorflow.org/) – deep learning framework  
- [PyTorch](https://pytorch.org/) – deep learning framework  
- [torchvision](https://pytorch.org/vision/stable/index.html) – computer vision datasets & models  
- [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) – EfficientNet implementation  

## 🚀 How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/ghstmd/skin-cancer-classification.git
   cd skin-cancer-classification
   ```
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Flask app
    ```bash
    python app.py
    ```
## 📊 Results
- Best Model: EfficientNetB0 (fine-tuned)
- Training with image + metadata improved performance compared to using images alone.
