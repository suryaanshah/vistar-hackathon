# Pneumonia Detection from Chest X-Rays using PyTorch & ResNet50

This project uses a deep learning model (ResNet50) trained with PyTorch to classify chest X-ray images as either showing signs of **Pneumonia** or being **Normal**. It includes scripts for data setup, model training, evaluation, and a simple web application interface built with Streamlit for inference.



---

## 🚀 Features

* **Image Classification:** Binary classification (Pneumonia vs. Normal).
* **Transfer Learning:** Utilizes a ResNet50 model pre-trained on ImageNet.
* **Data Augmentation:** Applies random cropping and flipping during training for robustness.
* **Evaluation Metrics:** Calculates Accuracy, Precision, Recall, F1-Score, and displays a Confusion Matrix.
* **Web Interface:** Simple Streamlit app for uploading images and getting predictions.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Deep Learning:** PyTorch, Torchvision
* **Data Handling:** NumPy, OpenCV (cv2)
* **Evaluation:** Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Web App:** Streamlit

---

## ⚙️ Setup (Local)

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    cd your_repository_name
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

## ▶️ Usage

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

## ☁️ Deployment

This application is deployed using Streamlit Community Cloud and is accessible at:

**(Optional: Add your Streamlit Cloud app URL here if you deployed it)**

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
    * **PyTorch:** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, *32*. ([https://pytorch.org/](https://pytorch.org/))
    * **Torchvision:** Marcel, S. and Rodriguez, Y. (2010). Torchvision the machine-vision package of Torch. *In Proceedings of the 18th ACM international conference on Multimedia* (pp. 1485-1488). ([https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html))
    * **Scikit-learn:** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, *12*, 2825-2830. ([https://scikit-learn.org/](https://scikit-learn.org/))
    * **Matplotlib:** Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in science & engineering*, *9*(3), 90-95. ([https://matplotlib.org/](https://matplotlib.org/))
    * **Seaborn:** Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, *6*(60), 3021. ([https://seaborn.pydata.org/](https://seaborn.pydata.org/))
    * **Streamlit:** Streamlit Inc. (2020). Streamlit: The fastest way to build custom ML tools. ([https://streamlit.io](https://streamlit.io))
    * **OpenCV:** Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*. ([https://opencv.org/](https://opencv.org/))

* **Tutorials/Code Structure:**
    * Code structure and training loops were adapted from standard PyTorch and Torchvision examples and tutorials available at [pytorch.org/tutorials](https://pytorch.org/tutorials).

---

## 📜 License

**(Optional: Add a license if you wish, e.g., MIT License)**