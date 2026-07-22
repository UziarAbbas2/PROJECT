# Pneumonia Detection using Chest X-Ray Images (Deep Learning + OpenCV)

Binary image classifier (NORMAL vs PNEUMONIA) built with TensorFlow/Keras,
OpenCV for image I/O and preprocessing, and scikit-learn for evaluation
metrics. Includes transfer-learning models (VGG16 / ResNet50), a custom
CNN, and Grad-CAM visualization to highlight the lung regions that drove
each prediction.

Dataset: [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Includes: a local training pipeline, a **Google Colab notebook** (train on a
free GPU, no local setup needed) and a **Flask web app** (upload an X-ray in
your browser, get a prediction + Grad-CAM heatmap).

## 1. Project structure

```
pneumonia-detection/
├── config.py                          # paths + hyperparameters
├── download_dataset.py                # optional: pulls the dataset via Kaggle API
├── requirements.txt
├── Pneumonia_Detection_ChestXray_Colab.ipynb   # <-- Google Colab notebook
├── src/
│   ├── data_preprocessing.py          # OpenCV-based image loading, Keras generators
│   ├── model.py                       # custom CNN + VGG16/ResNet50 transfer models
│   ├── train.py                       # training loop, callbacks, class weighting
│   ├── evaluate.py                    # confusion matrix, ROC curve, report
│   ├── predict.py                     # single-image inference + Grad-CAM
│   └── gradcam.py                     # Grad-CAM heatmap generation (OpenCV overlay)
├── flask_app/                         # <-- Flask web app
│   ├── app.py
│   ├── requirements.txt
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── static/
│       ├── css/style.css
│       └── uploads/                   # uploaded images + Grad-CAM outputs land here
├── models/                            # saved .h5 models land here
└── outputs/                           # plots, logs, reports land here
```

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Get the dataset

**Option A — manual:**
Download the zip from the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia),
extract it, and make sure you end up with this layout in the project root:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

**Option B — Kaggle API:**
```bash
pip install kaggle
# place your kaggle.json API token in ~/.kaggle/kaggle.json
python download_dataset.py
```

If your data lives somewhere else, set an environment variable instead of
editing `config.py`:
```bash
export PNEUMONIA_DATASET_DIR=/path/to/chest_xray
```

> Note: the Kaggle `val/` split only has 16 images. For more reliable
> validation you can optionally merge `val/` back into `train/` and carve
> out a proper stratified split — the pipeline works either way since it
> just reads whatever is in `TRAIN_DIR` / `VAL_DIR`.

## 4. Train

```bash
python -m src.train --model_type transfer_vgg16 --epochs 25 --batch_size 32
```

`--model_type` options: `custom_cnn`, `transfer_vgg16`, `transfer_resnet50`.

This will:
- Load data via `ImageDataGenerator` (augmentation on the train split only)
- Automatically compute class weights (the dataset is imbalanced — more
  PNEUMONIA images than NORMAL)
- Train with early stopping, LR reduction on plateau, and checkpointing
- Save the best model to `models/best_model.h5` and final model to
  `models/final_model.h5`
- Save accuracy/loss curves to `outputs/training_curves.png`

## 5. Evaluate

```bash
python -m src.evaluate --model_path models/best_model.h5
```

Produces in `outputs/`:
- `classification_report.txt` (precision/recall/F1 per class)
- `confusion_matrix.png`
- `roc_curve.png`

## 6. Predict on a single X-ray

```bash
python -m src.predict --image path/to/xray.jpg --model_path models/best_model.h5 --gradcam
```

Prints the predicted class + confidence, and (with `--gradcam`) saves a
heatmap overlay to `outputs/gradcam_<filename>` showing which regions of
the lung the model focused on.

## 7. Train on Google Colab (free GPU, no local setup)

1. Go to [colab.research.google.com](https://colab.research.google.com), choose
   **Upload**, and select `Pneumonia_Detection_ChestXray_Colab.ipynb` from this
   project.
2. `Runtime > Change runtime type` → select **GPU**.
3. Run the cells top to bottom:
   - Installs libraries
   - Prompts you to upload your Kaggle `kaggle.json` API token, then
     downloads + unzips the dataset straight into the Colab VM
   - Builds/trains a VGG16 transfer-learning model
   - Plots accuracy/loss curves, confusion matrix, ROC curve
   - Runs Grad-CAM on a sample X-ray using OpenCV
   - Mounts your Google Drive and saves `best_model.h5` there
4. Download `best_model.h5` from Drive and drop it into `models/` in this
   project (or point `flask_app` / `src/predict.py` at it) to use it locally.

## 8. Run the Flask web app

Lets you upload a chest X-ray in the browser and see the prediction +
Grad-CAM heatmap.

```bash
pip install -r flask_app/requirements.txt
# make sure a trained model exists at models/best_model.h5
# (either train locally with src/train.py, or drop in the one from Colab)
python flask_app/app.py
```

Then open **http://127.0.0.1:5000** in your browser, upload a `.jpg`/`.png`
chest X-ray, and click **Analyze X-Ray**. You'll see:
- The predicted class (NORMAL / PNEUMONIA) with confidence
- A Grad-CAM overlay highlighting the regions that influenced the prediction

To point the app at a different model file:
```bash
export PNEUMONIA_MODEL_PATH=/path/to/your_model.h5
python flask_app/app.py
```

## 9. How OpenCV is used

- `src/data_preprocessing.py`: `cv2.imread`, `cv2.cvtColor` (BGR→RGB),
  `cv2.resize`, and an optional `cv2.createCLAHE` contrast-enhancement step
  tailored to X-ray images.
- `src/gradcam.py`: `cv2.resize` to upsample the low-resolution CNN
  activation map, `cv2.applyColorMap` to turn it into a heatmap, and
  `cv2.addWeighted` to blend it with the original X-ray.
- `src/predict.py`: `cv2.imread` / `cv2.imwrite` for reading the input
  image and saving the Grad-CAM overlay.

## 10. Notes & tips

- Start with `transfer_vgg16` — transfer learning converges much faster
  and typically performs better than a from-scratch CNN on a dataset this
  size (~5,800 images).
- If you have a GPU, TensorFlow will use it automatically; training the
  custom CNN on CPU for 25 epochs can take a while given ~5,200 training
  images.
- The dataset is imbalanced (~3,875 PNEUMONIA vs ~1,341 NORMAL in train);
  `train.py` already applies `class_weight="balanced"` to compensate.
- For deployment, wrap `src/predict.py`'s `predict_single_image()` function
  in a Flask/FastAPI endpoint or a Streamlit app.
