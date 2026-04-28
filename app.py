import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO

# --- Konfigurasi ---
NUM_CLASSES = 15

# Label kelas sayuran (sesuaikan dengan urutan kelas saat training)
CLASS_NAMES = [
    "Bean", "Bitter Gourd", "Bottle Gourd", "Brinjal", "Broccoli",
    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
]

# Emoji mapping untuk setiap sayuran
CLASS_EMOJI = {
    "Bean": "🫘", "Bitter Gourd": "🥒", "Bottle Gourd": "🍐", "Brinjal": "🍆",
    "Broccoli": "🥦", "Cabbage": "🥬", "Capsicum": "🫑", "Carrot": "🥕",
    "Cauliflower": "🥦", "Cucumber": "🥒", "Papaya": "🍈", "Potato": "🥔",
    "Pumpkin": "🎃", "Radish": "🥕", "Tomato": "🍅"
}

# --- Page Config ---
st.set_page_config(
    page_title="VeganTeng — Klasifikasi Sayuran",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS untuk tampilan premium ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f4c3a 0%, #1a7a5c 40%, #2ecc71 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(46, 204, 113, 0.15);
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    .hero-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Card container */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #0f4c3a 0%, #1a7a5c 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(46, 204, 113, 0.2);
    }
    .result-card .emoji {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    .result-card .label {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .result-card .confidence {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    /* Confidence bar */
    .conf-bar-bg {
        background: rgba(255,255,255,0.15);
        border-radius: 12px;
        height: 14px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .conf-bar-fill {
        background: linear-gradient(90deg, #2ecc71, #a3f7bf);
        height: 100%;
        border-radius: 12px;
        transition: width 0.6s ease;
    }

    /* Model info badges */
    .model-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1a7a5c, #2ecc71);
        color: #fff;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Top-5 table */
    .top5-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 6px;
    }
    .top5-table th {
        text-align: left;
        padding: 0.6rem 1rem;
        font-weight: 600;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .top5-table td {
        padding: 0.7rem 1rem;
        background: rgba(255,255,255,0.04);
        font-size: 0.95rem;
    }
    .top5-table tr td:first-child {
        border-radius: 10px 0 0 10px;
    }
    .top5-table tr td:last-child {
        border-radius: 0 10px 10px 0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a3326 0%, #0f4c3a 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #2ecc71 !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a7a5c 0%, #2ecc71 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(46, 204, 113, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(46, 204, 113, 0.45);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(46, 204, 113, 0.4);
        border-radius: 16px;
        padding: 1rem;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 12px;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Hero Header ---
st.markdown("""
<div class="hero-header">
    <h1>🥦 VeganTeng</h1>
    <p>Klasifikasi Sayuran</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🎛️ Pengaturan Model")
    st.markdown("---")

    model_option = st.selectbox(
        "Pilih Model Inferensi",
        (
            "YOLO11n",
            "YOLO26n",
            "MobileNetV3-Small",
            "EfficientNetV2-S",
            "ResNet18",
            "VGG11",
        ),
        help="Pilih arsitektur model yang ingin digunakan untuk inferensi."
    )

    st.markdown("---")
    st.markdown("### 📋 Info Model")

    model_info = {
        "YOLO11n": {"params": "~2.6M", "tipe": "Object Detection", "desc": "Ringan & cepat, cocok untuk deteksi real-time."},
        "YOLO26n": {"params": "~2.6M", "tipe": "Object Detection", "desc": "Versi terbaru YOLO untuk deteksi objek."},
        "MobileNetV3-Small": {"params": "~2.5M", "tipe": "Klasifikasi", "desc": "Ultra-ringan, optimal untuk perangkat mobile."},
        "EfficientNetV2-S": {"params": "~21.5M", "tipe": "Klasifikasi", "desc": "Keseimbangan antara akurasi dan kecepatan."},
        "ResNet18": {"params": "~11.7M", "tipe": "Klasifikasi", "desc": "Arsitektur klasik dengan residual connections."},
        "VGG11": {"params": "~132.9M", "tipe": "Klasifikasi", "desc": "Arsitektur deep sederhana, akurasi tinggi."},
    }

    # Cari info model yang dipilih
    selected_key = model_option
    info = model_info.get(selected_key, {})

    if info:
        st.markdown(f"""
        <div class="glass-card">
            <span class="model-badge">{info['tipe']}</span>
            <p style="margin-top: 0.8rem; font-size: 0.95rem;">{info['desc']}</p>
            <p style="font-size: 0.85rem; opacity: 0.6;">Parameter: {info['params']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-size:0.75rem; opacity:0.4;'>VeganTeng — Computer Vision Project</p>",
        unsafe_allow_html=True
    )


# --- Fungsi Load Model dengan Cache ---
@st.cache_resource
def load_model(model_name):
    if "YOLO11n" in model_name:
        return YOLO("yolo11n.pt")

    elif "YOLO26n" in model_name:
        return YOLO("yolo26n.pt")

    elif "MobileNetV3" in model_name:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
        model.load_state_dict(torch.load("best_mobilenet_v3_small.pth", map_location="cpu"))

    elif "EfficientNetV2" in model_name:
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        model.load_state_dict(torch.load("best_efficientnet_v2_s.pth", map_location="cpu"))

    elif "ResNet18" in model_name:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load("best_resnet18.pth", map_location="cpu"))

    elif "VGG11" in model_name:
        model = models.vgg11(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
        model.load_state_dict(torch.load("best_vgg11.pth", map_location="cpu"))

    else:
        return None

    model.eval()
    return model


# --- Load Model ---
model = load_model(model_option)

# --- Area Utama ---
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Unggah Gambar")
    uploaded_file = st.file_uploader(
        "Seret & lepas gambar sayuran di sini",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

        process_btn = st.button("🔍 Proses Gambar", use_container_width=True)
    else:
        st.markdown("""
        <div style="
            border: 2px dashed rgba(46, 204, 113, 0.3);
            border-radius: 16px;
            padding: 4rem 2rem;
            text-align: center;
            margin-top: 1rem;
        ">
            <p style="font-size: 3rem; margin-bottom: 0.5rem;">📷</p>
            <p style="font-size: 1rem; opacity: 0.6;">Unggah gambar sayuran untuk memulai klasifikasi</p>
            <p style="font-size: 0.8rem; opacity: 0.4;">Format: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        process_btn = False

with col_result:
    st.markdown("### 📊 Hasil Analisis")

    if uploaded_file is not None and model is not None and process_btn:
        with st.spinner("⏳ Memproses gambar..."):
            if "YOLO" in model_option:
                # --- YOLO Object Detection ---
                results = model(image)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Hasil Deteksi Objek", use_container_width=True)

                # Tampilkan jumlah objek terdeteksi
                num_detections = len(results[0].boxes)
                st.markdown(f"""
                <div class="result-card">
                    <div class="emoji">🔍</div>
                    <div class="label">{num_detections} Objek Terdeteksi</div>
                    <div class="confidence">Model: {model_option}</div>
                </div>
                """, unsafe_allow_html=True)

                # Detail deteksi jika ada
                if num_detections > 0:
                    st.markdown("#### Daftar Objek Terdeteksi")
                    for i, box in enumerate(results[0].boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = results[0].names.get(cls_id, f"Kelas {cls_id}")
                        emoji = CLASS_EMOJI.get(cls_name, "🌿")
                        conf_pct_det = conf * 100
                        st.markdown(f"{emoji} **{cls_name}** — `{conf_pct_det:.1f}%`")

            else:
                # --- Klasifikasi CNN ---
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

                pred_idx = predicted.item()
                pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Kelas {pred_idx}"
                pred_emoji = CLASS_EMOJI.get(pred_label, "🌿")
                conf_val = confidence.item()
                conf_pct = conf_val * 100

                # Kartu hasil utama
                st.markdown(f"""
                <div class="result-card">
                    <div class="emoji">{pred_emoji}</div>
                    <div class="label">{pred_label}</div>
                    <div class="confidence">Confidence: {conf_pct:.1f}%</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width: {conf_pct:.1f}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Top-5 Prediksi
                st.markdown("#### 🏆 Top-5 Prediksi")
                top5_probs, top5_indices = torch.topk(probabilities, 5)

                for rank, (prob, idx) in enumerate(zip(top5_probs, top5_indices), 1):
                    idx_val = idx.item()
                    label = CLASS_NAMES[idx_val] if idx_val < len(CLASS_NAMES) else f"Kelas {idx_val}"
                    emoji = CLASS_EMOJI.get(label, "🌿")
                    prob_pct = prob.item() * 100
                    bar_width = prob.item() * 100

                    if rank == 1:
                        bg = "rgba(46,204,113,0.15)"
                        border = "1px solid rgba(46,204,113,0.3)"
                    else:
                        bg = "rgba(255,255,255,0.03)"
                        border = "1px solid rgba(255,255,255,0.06)"

                    st.markdown(f"""<div style="background:{bg}; border:{border}; border-radius:12px; padding:0.8rem 1.2rem; margin-bottom:0.5rem;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
<span style="font-weight:600;">#{rank} {emoji} {label}</span>
<span style="font-weight:500;">{prob_pct:.1f}%</span>
</div>
<div style="background:rgba(255,255,255,0.08); border-radius:6px; height:8px; overflow:hidden;">
<div style="width:{bar_width:.1f}%; height:100%; background:linear-gradient(90deg,#2ecc71,#a3f7bf); border-radius:6px;"></div>
</div>
</div>""", unsafe_allow_html=True)

    elif uploaded_file is None:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 4rem 2rem;
            opacity: 0.4;
        ">
            <p style="font-size: 3rem;">📊</p>
            <p>Hasil analisis akan tampil di sini</p>
        </div>
        """, unsafe_allow_html=True)

    elif model is None:
        st.error("⚠️ Model tidak tersedia. Silakan pilih model lain.")