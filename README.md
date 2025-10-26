# Pneumonia Detection from Chest X-Rays using PyTorch & ResNet50

This project uses a deep learning model (ResNet50) trained with PyTorch to classify chest X-ray images as either showing signs of **Pneumonia** or being **Normal**. It includes scripts for data setup, model training, evaluation, and a simple web application interface built with Streamlit for inference.



---

## Features

* **Image Classification:** Binary classification (Pneumonia vs. Normal).
* **Transfer Learning:** Utilizes a ResNet50 model pre-trained on ImageNet.
* **Data Augmentation:** Applies random cropping and flipping during training for robustness.
* **Evaluation Metrics:** Calculates Accuracy, Precision, Recall, F1-Score, and displays a Confusion Matrix.
* **Web Interface:** Simple Streamlit app for uploading images and getting predictions.

---

## Tech Stack

* **Language:** Python 3.9+
* **Deep Learning:** PyTorch, Torchvision
* **Data Handling:** NumPy, OpenCV (cv2)
* **Evaluation:** Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Web App:** Streamlit

---

## Setup (Local)

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/suryaanshah/vistar-hackathon.git](https://github.com/suryaanshah/vistar-hackathon.git)
    cd vistar-hackathon
    ```

2.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *Note: If you plan to use a GPU locally, ensure you have the correct NVIDIA drivers, CUDA Toolkit, and cuDNN installed, and install the GPU-enabled version of PyTorch.*

4.  **Download Data:**
    * Download the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    * Unzip the `archive.zip` file.
    * Ensure the resulting `chest_xray` folder is in the root of the project directory.

---

## Usage

1.  **Data Exploration (Optional):**
    * Inspect the dataset structure and view sample images.
    ```bash
    python explore.py
    ```

2.  **Train the Model:**
    * This script will load the data, apply augmentations, train the ResNet50 model (fine-tuning the last layer), and save the trained weights to `pneumonia_classifier_model.pth`.
    ```bash
    python train.py
    ```
    * *Requires a GPU for reasonable training time.*

3.  **Evaluate the Model:**
    * This script loads the saved model and evaluates its performance on the unseen test set, printing a classification report and showing a confusion matrix.
    ```bash
    python evaluate.py
    ```

4.  **Run the Web App:**
    * This launches the Streamlit application locally. Your browser should open automatically.
    ```bash
    streamlit run app.py
    ```
    * You can then upload chest X-ray images to get predictions.

---

## Deployment

This application is deployed using Streamlit Community Cloud and is accessible at: [Streamlit](https://vistar-hackathon-arp2ylkx8zrnrpy2nbcvcu.streamlit.app/)

The deployment uses the code from the `main` branch of this GitHub repository and installs dependencies from `requirements.txt`.

---

## 🙏 Citations & Acknowledgements

This project utilizes several open-source libraries and datasets.

* **Dataset:**
    * Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, V3, doi: 10.17632/rscbjbr9sj.3
        * Dataset link on Kaggle: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

* **Model Architecture (ResNet):**
    * He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *In Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778). ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))

* **Libraries:**
    * PyTorch
    * Torchvision
    * Scikit-learn
    * Matplotlib
    * Seaborn
    * Streamlit
    * OpenCV
    * NumPy
    * Pillow

* **Tutorials/Code Structure:**
    * Code structure and training loops were adapted from standard PyTorch and Torchvision examples, particularly inspired by:
        * **Transfer Learning Tutorial:** [pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
        * **Datasets & DataLoaders:** [pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
        * **Training a Classifier:** [pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
---
## 📜 License

Unlicenced