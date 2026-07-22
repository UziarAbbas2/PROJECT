import os
import time
import uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

app = Flask(__name__)

# Ensure absolute paths relative to app.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'temp', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'temp', 'results')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_garbage_unet.h5')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Model inputs dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256

def double_conv_block(x, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(x, num_filters):
    x = double_conv_block(x, num_filters)
    p = layers.MaxPool2D(pool_size=(2, 2))(x)
    return x, p

def decoder_block(x, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip_features])
    x = double_conv_block(x, num_filters)
    return x

def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)
    
    b1 = double_conv_block(p4, 512)
    
    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d4)
    model = models.Model(inputs, outputs, name='U-Net')
    return model

# Load the pretrained U-Net Model
print(f"Building U-Net Model and loading weights from {MODEL_PATH}...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
    model = build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print("Pretrained model weights loaded successfully!")
    else:
        print(f"Warning: {MODEL_PATH} not found. Running with initialized weights.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    import traceback
    traceback.print_exc()
    model = None

def process_image(img_path):
    """
    Load image, preprocess it for the model, make prediction,
    and generate high-resolution binary mask and overlay.
    """
    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        try:
            pil_img = Image.open(img_path).convert('RGB')
            orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as err:
            raise ValueError(f"Could not decode image file: {err}")
    
    orig_height, orig_width = orig_bgr.shape[:2]
    
    # Preprocess image for U-Net (convert to RGB, resize, normalize)
    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_input = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0) # Shape: (1, 256, 256, 3)
    
    # Run prediction
    start_time = time.time()
    if model is not None:
        pred = model.predict(img_input, verbose=0)[0]  # Shape: (256, 256, 1)
    else:
        pred = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    inference_time = (time.time() - start_time) * 1000  # in ms
    
    # Threshold predictions to create binary mask (256, 256)
    mask_256 = (pred.squeeze() > 0.5).astype(np.uint8)
    
    # Calculate garbage coverage percentage
    garbage_pixels = int(np.sum(mask_256 == 1))
    total_pixels = int(mask_256.size)
    coverage_pct = float(garbage_pixels / total_pixels) * 100.0
    
    # Upscale the mask back to the original image resolution
    mask_orig = cv2.resize(mask_256, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
    
    # Save the original image in temp results directory
    unique_id = str(uuid.uuid4())
    orig_save_name = f"{unique_id}_orig.jpg"
    orig_save_path = os.path.join(RESULT_FOLDER, orig_save_name)
    cv2.imwrite(orig_save_path, orig_bgr)
    
    # Generate Visual Binary Mask (Black background, White garbage)
    mask_visual = mask_orig * 255
    mask_save_name = f"{unique_id}_mask.png"
    mask_save_path = os.path.join(RESULT_FOLDER, mask_save_name)
    cv2.imwrite(mask_save_path, mask_visual)
    
    # Generate Overlay (Original image + Glowing Violet/Purple overlay on garbage)
    overlay_color = np.array([246, 92, 139], dtype=np.float32)  # Glowing Purple/Violet BGR
    alpha = 0.45  # Transparency factor
    
    overlay = orig_bgr.copy().astype(np.float32)
    garbage_indices = (mask_orig == 1)
    
    if np.any(garbage_indices):
        blended = overlay[garbage_indices] * (1.0 - alpha) + overlay_color * alpha
        overlay[garbage_indices] = blended
        
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    overlay_save_name = f"{unique_id}_overlay.jpg"
    overlay_save_path = os.path.join(RESULT_FOLDER, overlay_save_name)
    cv2.imwrite(overlay_save_path, overlay)
    
    return {
        "original_url": f"/static/temp/results/{orig_save_name}",
        "mask_url": f"/static/temp/results/{mask_save_name}",
        "overlay_url": f"/static/temp/results/{overlay_save_name}",
        "coverage_percentage": round(coverage_pct, 2),
        "inference_time_ms": round(inference_time, 1),
        "resolution": f"{orig_width} x {orig_height}",
        "garbage_pixel_count": garbage_pixels,
        "total_pixel_count": total_pixels
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    try:
        ext = os.path.splitext(file.filename)[1]
        if not ext:
            ext = ".jpg"
        upload_name = f"{str(uuid.uuid4())}{ext}"
        upload_path = os.path.join(UPLOAD_FOLDER, upload_name)
        file.save(upload_path)
        
        # Process and run inference
        results = process_image(upload_path)
        
        # Clean up uploaded file
        try:
            if os.path.exists(upload_path):
                os.remove(upload_path)
        except Exception as cleanup_err:
            print(f"Error deleting uploaded temp file: {cleanup_err}")
            
        return jsonify(results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask Application on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)
