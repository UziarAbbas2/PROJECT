"""
Central configuration for the Pneumonia Detection project.
Edit the paths below to match where you extracted the Kaggle dataset:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
# After downloading + unzipping the Kaggle dataset you should have:
# DATASET_DIR/
#     chest_xray/
#         train/
#             NORMAL/
#             PNEUMONIA/
#         val/
#             NORMAL/
#             PNEUMONIA/
#         test/
#             NORMAL/
#             PNEUMONIA/
DATASET_DIR = os.environ.get("PNEUMONIA_DATASET_DIR", "./chest_xray")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")

# ----------------------------------------------------------------------
# Image / training hyperparameters
# ----------------------------------------------------------------------
IMG_SIZE = 224                 # images resized to IMG_SIZE x IMG_SIZE
CHANNELS = 3                   # 3 for transfer-learning backbones (RGB)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
SEED = 42

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Which architecture to train: "custom_cnn" or "transfer_vgg16" or "transfer_resnet50"
MODEL_TYPE = "transfer_vgg16"
