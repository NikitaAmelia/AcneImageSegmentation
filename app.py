from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.color import rgb2lab
from skimage.segmentation import slic, mark_boundaries
from PIL import Image


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Acne Segmentation",
    page_icon="🔬",
    layout="wide",
)

# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT PATH
# ============================================================
# Struktur folder yang digunakan:
#
# Acne-Segmentation-Streamlit/
# ├── app.py
# ├── requirements.txt
# ├── Models/
# │   ├── superpixel_unet.pth
# │   ├── superpixel_unetplusplus.pth
# │   ├── cam_unet.pth
# │   ├── cam_unetplusplus.pth
# │   └── resnet34_classifier.pth
# └── AcneDataset/
#     ├── Blackheads/
#     ├── Whiteheads/
#     ├── Papules/
#     ├── Pustules/
#     └── Cyst/
#
# AcneDataset tidak diperlukan untuk inference gambar upload.
# Folder tersebut disimpan jika nanti aplikasi dikembangkan
# untuk evaluasi terhadap ground-truth dataset.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
DATASET_DIR = BASE_DIR / "AcneDataset"

IMG_SIZE = 256
PRED_THRESHOLD = 0.50
CAM_THRESHOLD = 0.40

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

# Urutan ini mengikuti CLASS_NAMES pada notebook CAM
# saat dataset/classifier disiapkan.
CLASS_NAMES = [
    "Blackheads",
    "Whiteheads",
    "Papules",
    "Pustules",
    "Cyst",
]


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATHS = {
    "Superpixel": {
        "U-Net": MODELS_DIR / "superpixel_unet.pth",
        "U-Net++": MODELS_DIR / "superpixel_unetplusplus.pth",
    },
    "CAM": {
        "U-Net": MODELS_DIR / "cam_unet.pth",
        "U-Net++": MODELS_DIR / "cam_unetplusplus.pth",
    },
}

CLASSIFIER_PATH = MODELS_DIR / "resnet34_classifier.pth"


# ============================================================
# INFERENCE TRANSFORM
# Sama dengan validation/test transform pada notebook:
# Resize -> Normalize -> ToTensorV2
# ============================================================

INFERENCE_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])


# ============================================================
# HELPER: LOAD STATE DICT
# ============================================================

def load_state_dict(path):
    """Load state_dict yang disimpan menggunakan torch.save()."""
    state = torch.load(path, map_location=DEVICE)

    # Notebook menyimpan langsung model.state_dict().
    # Bagian ini juga membuat aplikasi lebih aman jika file
    # suatu saat disimpan sebagai {'state_dict': ...}.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    return state


# ============================================================
# LOAD SEGMENTATION MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_segmentation_model(method, architecture):
    if architecture == "U-Net":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
    elif architecture == "U-Net++":
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
    else:
        raise ValueError(f"Arsitektur tidak dikenal: {architecture}")

    path = MODEL_PATHS[method][architecture]
    state_dict = load_state_dict(path)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# LOAD RESNET34 CLASSIFIER FOR CAM
# ============================================================

@st.cache_resource(show_spinner=False)
def load_cam_classifier():
    classifier = models.resnet34(weights=None)

    classifier.fc = nn.Linear(
        classifier.fc.in_features,
        len(CLASS_NAMES),
    )

    classifier.load_state_dict(
        load_state_dict(CLASSIFIER_PATH)
    )

    classifier = classifier.to(DEVICE)
    classifier.eval()

    return classifier


# ============================================================
# SUPERPIXEL PSEUDO MASK
# Mengikuti notebook Superpixel U-Net/U-Net++:
# SLIC -> RGB to LAB -> A threshold -> morphology
# -> remove small connected components.
# ============================================================

def make_superpixel_mask(img_tensor):
    """
    Parameters
    ----------
    img_tensor : torch.Tensor
        Tensor [C,H,W] yang sudah ImageNet-normalized.

    Returns
    -------
    clean_mask : np.ndarray
        Binary pseudo mask [H,W].
    segments : np.ndarray
        Label superpixel [H,W].
    """

    img = (
        img_tensor
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )

    # Denormalize
    img = img * STD + MEAN
    img = np.clip(img, 0, 1)

    # SLIC -- sama dengan notebook
    segments = slic(
        img,
        n_segments=250,
        compactness=10,
        sigma=1,
        start_label=1,
    )

    # RGB -> LAB
    lab = rgb2lab(img)

    mask = np.zeros(
        img.shape[:2],
        dtype=np.uint8,
    )

    # Threshold channel A -- sama dengan notebook
    A_THRESHOLD = 12

    for seg_id in np.unique(segments):
        region = segments == seg_id
        _, A_channel, _ = lab[region].mean(axis=0)

        if A_channel > A_THRESHOLD:
            mask[region] = 1

    # Morphology -- sama dengan notebook
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Remove small objects -- area > 50 px
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 50:
            clean_mask[labels == i] = 1

    return clean_mask.astype(np.float32), segments


# ============================================================
# CAM
# Mengikuti notebook CAM:
# ResNet34 layer4 -> FC class weights -> ReLU -> normalize
# -> resize 256x256 -> threshold 0.40.
# ============================================================

def generate_cam(img_tensor, classifier):
    feature_maps = []

    def forward_hook(module, inputs, output):
        feature_maps.clear()
        feature_maps.append(output.detach())

    hook = classifier.layer4.register_forward_hook(forward_hook)

    try:
        with torch.no_grad():
            output = classifier(
                img_tensor.unsqueeze(0).to(DEVICE)
            )

        pred_class = output.argmax(dim=1).item()

        if not feature_maps:
            raise RuntimeError(
                "Feature map layer4 tidak berhasil diperoleh untuk CAM."
            )

        feature = feature_maps[0].squeeze(0)

        weight = classifier.fc.weight[pred_class].detach()

        cam = torch.zeros(
            feature.shape[1:],
            device=DEVICE,
        )

        for i in range(feature.shape[0]):
            cam += weight[i] * feature[i]

        # ReLU
        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        # Normalize
        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        # Resize
        cam = cv2.resize(
            cam,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        # CAM pseudo mask -- threshold 0.40
        pseudo_mask = (
            cam > CAM_THRESHOLD
        ).astype(np.float32)

        return cam, pseudo_mask, pred_class

    finally:
        hook.remove()


# ============================================================
# CREATE CAM VISUALIZATION
# Notebook menggunakan 0.6 original + 0.4 heatmap.
# ============================================================

def create_cam_overlay(original, cam):
    heatmap = cv2.applyColorMap(
        np.uint8(cam * 255),
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    heatmap = heatmap.astype(np.float32) / 255.0

    overlay = (
        0.6 * original
        + 0.4 * heatmap
    )

    return np.clip(overlay, 0, 1)


# ============================================================
# SEGMENTATION PREDICTION
# ============================================================

def predict_mask(model, img_tensor):
    with torch.no_grad():
        output = model(
            img_tensor.unsqueeze(0).to(DEVICE)
        )

        probability = torch.sigmoid(output)

        pred_mask = (
            probability > PRED_THRESHOLD
        ).float()

    return (
        pred_mask
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


# ============================================================
# PREDICTION OVERLAY
# Sama dengan notebook: mask ditampilkan merah dengan alpha 0.45.
# ============================================================

def create_prediction_overlay(original, pred_mask):
    overlay = original.copy()
    alpha = 0.45

    red = np.zeros_like(original)
    red[..., 0] = 1.0

    overlay = np.where(
        pred_mask[..., None] == 1,
        (1 - alpha) * overlay + alpha * red,
        overlay,
    )

    return np.clip(overlay, 0, 1)


# ============================================================
# METRICS
# Mengikuti rumus notebook.
# Target = pseudo mask, karena notebook melakukan evaluasi
# menggunakan pseudo mask sebagai target segmentasi.
# ============================================================

def calculate_metrics(pred_mask, target_mask):
    pred = pred_mask.astype(np.float32)
    target = target_mask.astype(np.float32)

    intersection = np.sum(pred * target)

    dice = (
        2 * intersection
    ) / (
        np.sum(pred) + np.sum(target) + 1e-6
    )

    union = (
        np.sum(pred)
        + np.sum(target)
        - intersection
    )

    iou = intersection / (union + 1e-6)

    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))

    precision = tp / (tp + fp + 1e-6)

    fn = np.sum((1 - pred) * target)

    recall = tp / (tp + fn + 1e-6)

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
    }


# ============================================================
# CHECK MODEL FILES
# ============================================================

def get_missing_files():
    missing = []

    for method in MODEL_PATHS:
        for architecture in MODEL_PATHS[method]:
            path = MODEL_PATHS[method][architecture]
            if not path.exists():
                missing.append(path)

    if not CLASSIFIER_PATH.exists():
        missing.append(CLASSIFIER_PATH)

    return missing


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🔬 Acne Image Segmentation")
st.write(
    "Aplikasi segmentasi lesi jerawat menggunakan pseudo mask "
    "Superpixel atau CAM dengan model U-Net dan U-Net++"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Konfigurasi")

    method = st.selectbox(
        "Metode Pseudo Mask",
        ["Superpixel", "CAM"],
    )

    architecture = st.selectbox(
        "Model Segmentasi",
        ["U-Net", "U-Net++"],
    )

    st.divider()

    st.write(f"**Device:** `{DEVICE}`")
    st.write(f"**Input model:** `{IMG_SIZE} × {IMG_SIZE}`")
    st.write(f"**Prediction threshold:** `{PRED_THRESHOLD}`")

    if method == "Superpixel":
        st.write("**SLIC:** n_segments=250")
        st.write("**Compactness:** 10")
        st.write("**Sigma:** 1")
        st.write("**LAB A threshold:** 12")
        st.write("**Min. component area:** > 50 px")
    else:
        st.write("**Classifier:** ResNet34")
        st.write("**Feature layer:** layer4")
        st.write(f"**CAM threshold:** `{CAM_THRESHOLD}`")


# ============================================================
# MODEL STATUS
# ============================================================

missing_files = get_missing_files()

if missing_files:
    st.error(
        "File model belum lengkap. Pastikan struktur folder dan "
        "nama file .pth sesuai."
    )

    with st.expander("File yang belum ditemukan"):
        for path in missing_files:
            st.code(str(path))

    st.stop()


# ============================================================
# IMAGE INPUT: UPLOAD ATAU KAMERA
# ============================================================

st.subheader("📷 Input Gambar")

input_method = st.radio(
    "Pilih cara memasukkan gambar:",
    ["Upload Gambar", "Gunakan Kamera"],
    horizontal=True
)

image_file = None

if input_method == "Upload Gambar":
    image_file = st.file_uploader(
        "Upload gambar jerawat",
        type=["jpg", "jpeg", "png"],
        help="Pilih satu gambar jerawat dari perangkat."
    )

else:
    image_file = st.camera_input(
        "Ambil foto menggunakan kamera",
        help="Izinkan akses kamera jika browser memintanya, lalu ambil foto."
    )


# ============================================================
# INFERENCE
# ============================================================

if image_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    # Membaca gambar dari upload atau kamera
    pil_image = Image.open(image_file).convert("RGB")
    original_np = np.array(pil_image)

    # Sama dengan alur visualisasi notebook: image diproses pada 256x256.
    original_resized = cv2.resize(
        original_np,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    original_float = (
        original_resized.astype(np.float32) / 255.0
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    transformed = INFERENCE_TRANSFORM(
        image=original_resized
    )

    img_tensor = transformed["image"].float()

    # --------------------------------------------------------
    # LOAD SEGMENTATION MODEL
    # --------------------------------------------------------

    model_path = MODEL_PATHS[method][architecture]

    try:
        with st.spinner(
            f"Memuat model {method} + {architecture}..."
        ):
            seg_model = load_segmentation_model(
                method,
                architecture,
            )
    except Exception as e:
        st.error("Model segmentasi gagal dimuat.")
        st.exception(e)
        st.stop()

    # --------------------------------------------------------
    # GENERATE PSEUDO MASK
    # --------------------------------------------------------

    try:
        if method == "Superpixel":
            pseudo_mask, segments = make_superpixel_mask(
                img_tensor
            )

            # Sama dengan notebook: boundary merah.
            boundary_img = mark_boundaries(
                original_float,
                segments,
                color=(1, 0, 0),
            )

            cam = None
            cam_image = None
            predicted_class = None

        else:
            with st.spinner("Menghasilkan CAM dan CAM pseudo mask..."):
                classifier = load_cam_classifier()

                cam, pseudo_mask, predicted_class = generate_cam(
                    img_tensor,
                    classifier,
                )

            cam_image = create_cam_overlay(
                original_float,
                cam,
            )

            boundary_img = None

        # ----------------------------------------------------
        # SEGMENTATION PREDICTION
        # ----------------------------------------------------

        with st.spinner(
            f"Menjalankan {architecture}..."
        ):
            pred_mask = predict_mask(
                seg_model,
                img_tensor,
            )

        # Untuk visualisasi saja
        pred_mask_display = (pred_mask * 255).astype(np.uint8)

        # ----------------------------------------------------
        # PREDICTION OVERLAY
        # ----------------------------------------------------

        prediction_overlay = create_prediction_overlay(
            original_float,
            pred_mask,
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = calculate_metrics(
            pred_mask,
            pseudo_mask,
        )

    except Exception as e:
        st.error("Terjadi error saat proses inference.")
        st.exception(e)
        st.stop()

    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    st.success(
        f"Inference selesai — {method} + {architecture}"
    )

    if method == "CAM":
        st.info(
            "Kelas hasil klasifikasi CAM: "
            f"**{CLASS_NAMES[predicted_class]}**"
        )

    st.caption(
        f"Model: `{model_path.name}` | Device: `{DEVICE}`"
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    st.subheader("Hasil Visualisasi")

    cols = st.columns(5)

    if method == "Superpixel":
        with cols[0]:
            st.image(
                original_float,
                caption="Original",
                use_container_width=True,
            )

        with cols[1]:
            st.image(
                boundary_img,
                caption="Superpixel Boundary",
                use_container_width=True,
            )

        with cols[2]:
            st.image(
                pseudo_mask,
                caption="Superpixel Pseudo Mask",
                clamp=True,
                use_container_width=True,
            )

        with cols[3]:
            st.image(
                pred_mask_display,
                caption=f"{architecture} Prediction",
                clamp=True,
                use_container_width=True,
            )

        with cols[4]:
            st.image(
                prediction_overlay,
                caption="Prediction Overlay",
                use_container_width=True,
            )

    else:
        with cols[0]:
            st.image(
                original_float,
                caption="Original",
                use_container_width=True,
            )

        with cols[1]:
            st.image(
                cam_image,
                caption="CAM",
                use_container_width=True,
            )

        with cols[2]:
            st.image(
                pseudo_mask,
                caption="CAM Pseudo Mask",
                clamp=True,
                use_container_width=True,
            )

        with cols[3]:
            st.image(
                pred_mask_display,
                caption=f"{architecture} Prediction",
                clamp=True,
                use_container_width=True,
            )

        with cols[4]:
            st.image(
                prediction_overlay,
                caption="Prediction Overlay",
                use_container_width=True,
            )

    # ========================================================
    # METRICS
    # ========================================================

    st.subheader("Evaluasi Segmentasi")

    st.caption(
        "Nilai evaluasi menunjukkan tingkat kesesuaian hasil segmentasi terhadap pseudo mask yang digunakan sebagai target"
    )

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Dice",
        f"{metrics['Dice']:.4f}",
    )

    metric_cols[1].metric(
        "IoU",
        f"{metrics['IoU']:.4f}",
    )

    metric_cols[2].metric(
        "Precision",
        f"{metrics['Precision']:.4f}",
    )

    metric_cols[3].metric(
        "Recall",
        f"{metrics['Recall']:.4f}",
    )

    # ========================================================
    # DETAIL
    # ========================================================

    with st.expander("Detail proses"):
        st.write(f"**Metode pseudo mask:** {method}")
        st.write(f"**Model segmentasi:** {architecture}")
        st.write(f"**File model:** `{model_path.name}`")
        st.write(f"**Device:** `{DEVICE}`")

        if method == "Superpixel":
            st.write("**SLIC:** n_segments=250, compactness=10, sigma=1")
            st.write("**LAB A threshold:** 12")
            st.write("**Minimum connected component:** > 50 px")
        else:
            st.write("**CAM classifier:** ResNet34")
            st.write("**Feature layer:** layer4")
            st.write(f"**CAM pseudo-mask threshold:** {CAM_THRESHOLD}")

        st.write(f"**Segmentation threshold:** {PRED_THRESHOLD}")
