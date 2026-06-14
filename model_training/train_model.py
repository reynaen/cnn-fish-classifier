"""
CNN Fish Classification - Training Script
Dataset: A Large Scale Fish Dataset (lokal)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────
# 1. PATH DATASET LOKAL
# ──────────────────────────────────────────────
# Folder Fish_Dataset sudah dipindah ke dalam cnn-fish-classifier/
# Jalankan script ini dari folder: cnn-fish-classifier/model_training/
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'Fish_Dataset')
DATASET_DIR = os.path.abspath(DATASET_DIR)

if not os.path.exists(DATASET_DIR):
    print(f"[ERROR] Folder dataset tidak ditemukan di: {DATASET_DIR}")
    print("Pastikan folder Fish_Dataset sudah dipindah ke dalam cnn-fish-classifier/")
    exit(1)

classes = sorted([d for d in os.listdir(DATASET_DIR)
                  if os.path.isdir(os.path.join(DATASET_DIR, d))
                  and not d.startswith('.')])

print(f"Dataset ditemukan di : {DATASET_DIR}")
print(f"Kelas ({len(classes)}): {classes}")

NUM_CLASSES = len(classes)

# ──────────────────────────────────────────────
# 2. KONFIGURASI
# ──────────────────────────────────────────────
IMG_SIZE   = (64, 64)
BATCH_SIZE = 32
EPOCHS     = 15

# ──────────────────────────────────────────────
# 3. DATA GENERATOR + AUGMENTASI
# ──────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nData training : {train_gen.samples} gambar")
print(f"Data validasi : {val_gen.samples} gambar")

# Simpan class names untuk Flask app
class_names = list(train_gen.class_indices.keys())
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..')
np.save(os.path.join(OUTPUT_DIR, 'class_names.npy'), class_names)
print(f"Class names   : {class_names}")

# ──────────────────────────────────────────────
# 4. ARSITEKTUR MODEL CNN
# ──────────────────────────────────────────────
model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', padding='same',
                  input_shape=(64, 64, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.4),

    # Fully Connected
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ──────────────────────────────────────────────
# 5. TRAINING
# ──────────────────────────────────────────────
STATIC_DIR = os.path.join(OUTPUT_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'fish_cnn_model.keras'),
        monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
]

print("\nMemulai training...\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ──────────────────────────────────────────────
# 6. EVALUASI
# ──────────────────────────────────────────────
test_loss, test_acc = model.evaluate(val_gen, verbose=0)
print(f"\nAkurasi Validasi : {test_acc*100:.2f}%")
print(f"Loss Validasi    : {test_loss:.4f}")

# ──────────────────────────────────────────────
# 7. PLOT AKURASI & LOSS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train', color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='Validasi', color='orange')
axes[0].set_title('Kurva Akurasi Model CNN')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Akurasi')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='steelblue')
axes[1].plot(history.history['val_loss'], label='Validasi', color='orange')
axes[1].set_title('Kurva Loss Model CNN')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Grafik disimpan: static/training_history.png")

# ──────────────────────────────────────────────
# 8. CONFUSION MATRIX
# ──────────────────────────────────────────────
y_pred_probs = model.predict(val_gen, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = val_gen.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Klasifikasi Jenis Ikan')
plt.xlabel('Prediksi')
plt.ylabel('Label Asli')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix disimpan: static/confusion_matrix.png")

# ──────────────────────────────────────────────
# 9. CLASSIFICATION REPORT
# ──────────────────────────────────────────────
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n✅ Training selesai!")
print(f"   Model  : fish_cnn_model.keras")
print(f"   Kelas  : class_names.npy")
print(f"   Akurasi: {test_acc*100:.2f}%")
