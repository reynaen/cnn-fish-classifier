"""
Flask Web App - Klasifikasi Jenis Ikan dengan CNN
Tugas 10 - Praktikum Kecerdasan Buatan
"""

import os
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf

# ──────────────────────────────────────────────
# INISIALISASI APP
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Maks 10MB
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ──────────────────────────────────────────────
# LOAD MODEL & CLASS NAMES
# ──────────────────────────────────────────────
MODEL_PATH = 'fish_cnn_model.keras'
CLASSES_PATH = 'class_names.npy'

model = None
class_names = []

def load_model_and_classes():
    global model, class_names
    if os.path.exists(MODEL_PATH):
        print("Memuat model CNN...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model berhasil dimuat!")
    else:
        print(f"[PERINGATAN] File model '{MODEL_PATH}' tidak ditemukan.")
        print("Jalankan model_training/train_model.py terlebih dahulu.")

    if os.path.exists(CLASSES_PATH):
        class_names = list(np.load(CLASSES_PATH, allow_pickle=True))
        print(f"Kelas dimuat: {class_names}")
    else:
        # Fallback class names dari dataset ikan
        class_names = [
            "Black Sea Sprat", "Gilt-Head Bream", "Horse Mackerel",
            "Red Mullet", "Red Sea Bream", "Sea Bass",
            "Shrimp", "Striped Red Mullet", "Trout"
        ]
        print(f"Menggunakan class names default: {class_names}")

load_model_and_classes()

# ──────────────────────────────────────────────
# FUNGSI PREPROCESSING GAMBAR
# ──────────────────────────────────────────────
IMG_SIZE = (64, 64)

def preprocess_image(image_bytes):
    """Preprocess gambar dari bytes menjadi tensor siap prediksi."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # (1, 64, 64, 3)
    return img_array

# ──────────────────────────────────────────────
# ROUTE: HALAMAN UTAMA
# ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', class_names=class_names)

# ──────────────────────────────────────────────
# ROUTE: PREDIKSI
# ──────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400

    allowed_ext = {'jpg', 'jpeg', 'png', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed_ext:
        return jsonify({'error': f'Format file tidak didukung: .{ext}'}), 400

    if model is None:
        return jsonify({'error': 'Model belum dimuat. Jalankan training terlebih dahulu.'}), 503

    try:
        img_bytes = file.read()
        img_tensor = preprocess_image(img_bytes)

        # Prediksi
        predictions = model.predict(img_tensor, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx]) * 100

        # Top-3 prediksi
        top3_idx = np.argsort(predictions)[::-1][:3]
        top3 = [
            {
                'label': class_names[i],
                'confidence': round(float(predictions[i]) * 100, 2)
            }
            for i in top3_idx
        ]

        return jsonify({
            'predicted_class': class_names[predicted_idx],
            'confidence': round(confidence, 2),
            'top3': top3,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': f'Gagal memproses gambar: {str(e)}'}), 500

# ──────────────────────────────────────────────
# ROUTE: INFO MODEL
# ──────────────────────────────────────────────
@app.route('/model-info')
def model_info():
    if model is None:
        return jsonify({'status': 'Model belum dimuat'})

    total_params = model.count_params()
    layers_info = [
        {'name': l.name, 'type': type(l).__name__, 'output_shape': str(l.output_shape)}
        for l in model.layers
    ]
    return jsonify({
        'total_parameters': f"{total_params:,}",
        'num_classes': len(class_names),
        'class_names': class_names,
        'input_shape': str(model.input_shape),
        'layers': layers_info
    })

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
