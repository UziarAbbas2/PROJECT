# AI-Based Garbage Segmentation using Deep Learning

This is a complete end-to-end deep learning project for semantic segmentation of garbage/trash in images. The project utilizes a U-Net convolutional neural network architecture trained in TensorFlow/Keras to perform pixel-wise binary classification (garbage vs. background). It features a premium, interactive Flask-based web dashboard for local deployment and validation.

---

## 🚀 Key Features

*   **Deep Learning Segmentation**: Pixel-level binary segmentation of litter/garbage.
*   **Flask Web Deployment**: Interactive local server for processing images.
*   **Premium Glassmorphism UI**: High-end modern dark-theme user experience.
*   **Interactive Comparison Slider**: Swipe-based visual comparison between the original image and the segmented overlay.
*   **Garbage Coverage Metric**: Automatically calculates the percentage area covered by litter.
*   **Google Colab Training Notebook**: A complete, well-documented training pipeline (`garbage_segmentation.ipynb`) showing dataset download, model training, evaluation, and saving.

---

## 📁 Project Structure

```text
garbage_segmentation_project/
│
├── model/
│   └── best_garbage_unet.h5        # Pre-trained U-Net weights file (Copy here)
│
├── templates/
│   └── index.html                 # Flask dashboard frontend HTML template
│
├── static/
│   ├── css/
│   │   └── style.css              # Custom premium glassmorphic styling
│   ├── js/
│   │   └── main.js                # AJAX upload logic & interactive comparison slider
│   └── temp/                      # Directory for uploads and processed outputs (git ignored)
│
├── app.py                         # Flask backend deployment script
├── garbage_segmentation.ipynb     # Google Colab notebook for training the model from scratch
├── requirements.txt               # List of dependencies
└── README.md                      # This instruction manual
```

---

## 🛠️ Setup & Local Deployment

### Prerequisites
Make sure Python 3.10+ is installed on your system.

### 1. Installation
Clone or copy this folder to your local machine, open your terminal in this directory, and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Copy the Model
Ensure your pre-trained model file `best_garbage_unet.h5` is placed inside the `model/` folder:
*   Folder path: `garbage_segmentation_project/model/best_garbage_unet.h5`

### 3. Run the Flask Web Application
Launch the Flask development server:
```bash
python app.py
```

Open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## 📓 Google Colab Notebook (`garbage_segmentation.ipynb`)
If you want to train the model from scratch, open the `garbage_segmentation.ipynb` file in Google Colab. The notebook walks you through:
1.  **Dataset Acquisition**: Downloader script for garbage datasets.
2.  **Data Preprocessing**: Scaling, resizing to `256x256`, and binary mask normalization.
3.  **U-Net Model Definition**: Deep convolutional encoder-decoder network.
4.  **Custom Metrics**: Dice coefficient and Intersection-over-Union (IoU) metrics.
5.  **Training & Checkpoints**: Train with callbacks (Early Stopping, Learning Rate Decay).
6.  **Inference & Validation**: Visualizing segmented masks side-by-side with ground truths.
