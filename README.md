# 🧴 Acne Image Segmentation

A **Deep Learning-Based Acne Lesion Segmentation System** that delineates acne lesion regions from uploaded skin images using pseudo-mask generation methods and segmentation architectures. This project is developed as a **Machine Learning / Computer Vision research project** and provides an interactive **Streamlit web application**.

---

## 🔗 Live Demo (Streamlit App)

👉 [https://acneimagesegmentation.streamlit.app/]


---

## 🚀 Application Overview

The web application allows users to:

* Upload a skin image for acne lesion segmentation
* Select a pseudo-mask generation method: **Superpixel (SLIC)** or **CAM-Based**
* Select a segmentation architecture: **U-Net** or **U-Net++**
* View the input image, pseudo-mask, predicted segmentation mask, and prediction overlay
* Review available evaluation metrics, including **Dice Score, IoU, Precision, and Recall**, when provided by the application

---

## 🧠 Methodology

* **Pseudo-Mask Generation**
  * Superpixel segmentation using **Simple Linear Iterative Clustering (SLIC)**
  * **Class Activation Mapping (CAM)-Based** pseudo-mask generation
* **Semantic Segmentation Architectures**
  * **U-Net**
  * **U-Net++**
* **Encoder**
  * **ResNet34**
* **Evaluation Metrics**
  * Dice Score
  * Intersection over Union (IoU)
  * Precision
  * Recall

The project compares how different pseudo-mask generation methods perform as training targets for acne lesion segmentation using U-Net and U-Net++.

---

## 🛠 Tech Stack

* Python
* PyTorch
* Torchvision
* Segmentation Models PyTorch
* OpenCV
* NumPy
* Pillow
* scikit-image
* Matplotlib
* Streamlit

---

## 📁 Project Structure

```text
Acne-Segmentation-Streamlit/
│
├── app.py                       # Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Files and folders excluded from Git
└── Models/
    ├── superpixel_unet.pth
    ├── superpixel_unetplusplus.pth
    ├── cam_unet.pth
    ├── cam_unetplusplus.pth
    └── resnet34_classifier.pth
```

> The `Models/` directory contains the trained segmentation models and the classifier used by the CAM-based method. The dataset and Python virtual environment are not included in the repository.

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/NikitaAmelia/AcneImageSegmentation.git
cd AcneImageSegmentation
```

### 2. Create and activate a virtual environment (optional but recommended)

**Windows (CMD):**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The application should open in your browser. If it does not, open the local URL shown in the terminal.

---

## 📌 Notes

* This application performs inference on images uploaded by the user.
* The training dataset is not included in this repository.
* The trained model files in `Models/` are required for the application to run.
* Ensure that model filenames and paths in `app.py` match the files in the `Models/` directory.
* The application is intended for educational and research purposes and should not be used as a substitute for professional medical advice.

---

## 👩‍💻 Author

**Nikita Amelia Valencia**  
Machine Learning & Computer Vision

---

## 📜 License

This project is for educational and research purposes.
