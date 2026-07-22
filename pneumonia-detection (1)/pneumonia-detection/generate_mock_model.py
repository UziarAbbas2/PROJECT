"""
Generate a mock model (VGG16 architecture with random weights)
and save it as models/best_model.h5 so the Flask application can be tested immediately.
"""

import os
import sys

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers

def build_mock_vgg16(input_shape=(224, 224, 3)):
    print("Building VGG16 base model (untrained/random weights)...")
    # Using weights=None to avoid downloading ~50MB VGG16 weights during testing
    base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="transfer_vgg16")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )
    return model

def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model = build_mock_vgg16()
    
    print(f"Saving mock model to {config.BEST_MODEL_PATH}...")
    model.save(config.BEST_MODEL_PATH)
    print("Successfully generated and saved mock model!")

if __name__ == "__main__":
    main()
