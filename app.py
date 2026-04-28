import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO

# --- Konfigurasi ---
# GANTI ANGKA INI DENGAN JUMLAH KELAS SAYURAN ANDA SAAT TRAINING
NUM_CLASSES = 10 

st.title("Aplikasi Klasifikasi & Deteksi Sayuran")

# --- Pilihan 6 Model ---
model_option = st.selectbox(
    "Pilih Model Inferensi",
    (
        "1. YOLO11n (Object Detection)", 
        "2. MobileNetV3-Small", 
        "3. EfficientNetV2-S", 
        "4. ResNet18", 
        "5. VGG11",
        "6. Model Tambahan (Custom)" # Slot untuk model ke-6
    )
)

# --- Fungsi Load Model dengan Cache ---
# Gunakan cache agar model tidak di-load ulang setiap kali ada interaksi di web
@st.cache_resource
def load_model(model_name):
    if "YOLO11n" in model_name:
        return YOLO("yolo11n.pt")
        
    elif "MobileNetV3" in model_name:
        model = models.mobilenet_v3_small(weights=None)
        # Menyesuaikan layer terakhir MobileNetV3
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
        model.load_state_dict(torch.load("best_mobilenet_v3_small.pth", map_location='cpu'))
        
    elif "EfficientNetV2" in model_name:
        model = models.efficientnet_v2_s(weights=None)
        # Menyesuaikan layer terakhir EfficientNet
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        model.load_state_dict(torch.load("best_efficientnet_v2_s.pth", map_location='cpu'))
        
    elif "ResNet18" in model_name:
        model = models.resnet18(weights=None)
        # Menyesuaikan layer terakhir ResNet
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load("best_resnet18.pth", map_location='cpu'))
        
    elif "VGG11" in model_name:
        model = models.vgg11(weights=None)
        # Menyesuaikan layer terakhir VGG
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
        model.load_state_dict(torch.load("best_vgg11.pth", map_location='cpu'))
        
    elif "Model Tambahan" in model_name:
        # Template untuk model ke-6 Anda nantinya
        # model = ... 
        return None

    model.eval()
    return model

model = load_model(model_option)

# --- Upload & Proses Gambar ---
uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    if st.button("Proses Gambar"):
        if "YOLO" in model_option:
            results = model(image)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption='Hasil Deteksi', use_column_width=True)
        else:
            # Standar transformasi ImageNet
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)

            st.success(f"Hasil Prediksi: Kelas ke-{predicted.item()} (Confidence: {confidence.item():.2f})")