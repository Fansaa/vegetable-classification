<div align="center">

# 🥦 VeganTeng

### Klasifikasi & Deteksi Sayuran berbasis Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vegetable-classification.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Aplikasi web interaktif untuk mengklasifikasikan dan mendeteksi 15 jenis sayuran menggunakan berbagai arsitektur Deep Learning.

</div>

---

## 📋 Daftar Isi

- [Fitur](#-fitur)
- [Kelas Sayuran](#-kelas-sayuran)
- [Arsitektur Model](#-arsitektur-model)
- [Demo](#-demo)
- [Instalasi Lokal](#-instalasi-lokal)
- [Struktur Proyek](#-struktur-proyek)
- [Teknologi](#-teknologi)

---

## ✨ Fitur

- 🏷️ **Image Classification** — Klasifikasi gambar sayuran ke dalam 15 kelas menggunakan 6 arsitektur model
- 📊 **Top-5 Prediksi** — Menampilkan 5 kelas teratas beserta confidence score dan progress bar visual
- 🏛️ **6 Arsitektur Model** — Bandingkan performa YOLO, MobileNet, EfficientNet, ResNet, dan VGG
- 🎨 **UI Premium** — Tampilan modern dengan dark theme, glassmorphism, dan animasi halus
- ☁️ **Cloud Deployment** — Dideploy di Streamlit Cloud, akses kapan saja via browser

---

## 🥬 Kelas Sayuran

Aplikasi ini dapat mengenali **15 jenis sayuran**:

| # | Sayuran | Emoji | # | Sayuran | Emoji |
|---|---------|-------|---|---------|-------|
| 1 | Bean | 🫘 | 9 | Cauliflower | 🥦 |
| 2 | Bitter Gourd | 🥒 | 10 | Cucumber | 🥒 |
| 3 | Bottle Gourd | 🍐 | 11 | Papaya | 🍈 |
| 4 | Brinjal | 🍆 | 12 | Potato | 🥔 |
| 5 | Broccoli | 🥦 | 13 | Pumpkin | 🎃 |
| 6 | Cabbage | 🥬 | 14 | Radish | 🥕 |
| 7 | Capsicum | 🫑 | 15 | Tomato | 🍅 |
| 8 | Carrot | 🥕 | | | |

---

## 🧠 Arsitektur Model

| Model | Parameter | Deskripsi |
|-------|-----------|----------|
| **YOLOv8n-cls** | ~2.7M | YOLO v8 Nano untuk klasifikasi gambar, ringan & cepat |
| **YOLO26n-cls** | ~2.7M | YOLO v26 Nano untuk klasifikasi gambar, versi terbaru |
| **MobileNetV3-Small** | ~2.5M | Ultra-ringan, optimal untuk perangkat mobile |
| **EfficientNetV2-S** | ~21.5M | Keseimbangan antara akurasi dan kecepatan |
| **ResNet18** | ~11.7M | Arsitektur klasik dengan residual connections |
| **VGG11** | ~132.9M | Arsitektur deep sederhana, akurasi tinggi |

---

## 🚀 Demo

Akses aplikasi langsung di browser:

🔗 **[vegetable-classification.streamlit.app](https://vegetable-classification.streamlit.app)**

### Cara Penggunaan

1. Pilih model inferensi dari sidebar (YOLO untuk deteksi, atau CNN untuk klasifikasi)
2. Unggah gambar sayuran (format: JPG, JPEG, PNG)
3. Klik tombol **"🔍 Proses Gambar"**
4. Lihat hasil prediksi beserta confidence score

---

## 💻 Instalasi Lokal

### Prasyarat
- Python 3.10+
- pip

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/Fansaa/vegetable-classification.git
cd vegetable-classification

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

---

## 📁 Struktur Proyek

```
vegetable-classification/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (Streamlit Cloud)
├── README.md                       # Dokumentasi proyek
├── best_mobilenet_v3_small.pth     # Pretrained MobileNetV3-Small weights
├── best_efficientnet_v2_s.pth      # Pretrained EfficientNetV2-S weights
├── best_resnet18.pth               # Pretrained ResNet18 weights
├── best_vgg11.pth                  # Pretrained VGG11 weights
├── yolov8n-cls.pt                  # Pretrained YOLOv8n classification weights
└── yolo26n-cls.pt                  # Pretrained YOLO26n classification weights
```

---

## 🛠️ Teknologi

<div align="center">

| Kategori | Teknologi |
|----------|-----------|
| **Framework Web** | Streamlit |
| **Deep Learning** | PyTorch, Torchvision |
| **Object Detection** | Ultralytics (YOLO) |
| **Image Processing** | Pillow, OpenCV |
| **Deployment** | Streamlit Cloud |

</div>

---

<div align="center">

**VeganTeng** — Computer Vision Project

Made with ❤️ and 🥦

</div>
