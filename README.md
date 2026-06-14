# 🐟 FishCNN — Klasifikasi Jenis Ikan dengan CNN

**Tugas 10 - Praktikum Kecerdasan Buatan**  
Teknik Informatika, Universitas Bale Bandung

---

## 📌 Deskripsi

Aplikasi web berbasis **Flask** yang menggunakan **Convolutional Neural Network (CNN)** untuk mengklasifikasikan jenis ikan dari citra gambar. Model dilatih menggunakan dataset _A Large Scale Fish Dataset_ dari Kaggle.

### Kelas yang Dapat Dideteksi (9 Kelas)
| No | Nama Ikan |
|----|-----------|
| 1 | Black Sea Sprat |
| 2 | Gilt-Head Bream |
| 3 | Horse Mackerel |
| 4 | Red Mullet |
| 5 | Red Sea Bream |
| 6 | Sea Bass |
| 7 | Shrimp |
| 8 | Striped Red Mullet |
| 9 | Trout |

---

## 🏗️ Arsitektur CNN

```
Input (64×64×3)
    ↓
[Block 1] Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
[Block 2] Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
[Block 3] Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.4)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(9) + Softmax → Output
```

---

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/username/cnn-fish-classifier.git
cd cnn-fish-classifier
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Training Model
```bash
cd model_training
python train_model.py
cd ..
```

### 5. Jalankan Aplikasi
```bash
python app.py
```
Buka browser: `http://localhost:5000`

---

## ☁️ Deployment (Railway)

1. Push ke GitHub
2. Buka [railway.app](https://railway.app), klik **New Project → Deploy from GitHub**
3. Pilih repository ini
4. Railway otomatis mendeteksi `Procfile` dan deploy

---

## 🛠️ Teknologi

| Komponen | Teknologi |
|----------|-----------|
| Model AI | TensorFlow / Keras |
| Web Framework | Flask |
| Frontend | Bootstrap 5 + FontAwesome |
| Deployment | Railway / Render |
| Dataset | Kaggle (kagglehub) |

---

## 📄 Laporan

Laporan penelitian tersedia dalam folder `laporan/` dalam format DOCX dan PDF.

---

## 📹 Demo

Link YouTube: [Coming Soon]

---

*© 2025 - Praktikum Kecerdasan Buatan, Universitas Bale Bandung*
