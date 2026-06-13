import os
import io
import base64
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DISTANCES = [1]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
LEVELS = 16
IMG_SIZE = (128, 128)

MODEL_PATH = "decision_tree_model.pkl"

# Load model hasil training
model = joblib.load(MODEL_PATH)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # Resize ROI
    img = img.resize(IMG_SIZE, Image.LANCZOS)

    arr = np.array(img, dtype=np.uint8)

    # Kuantisasi 256 → 16 level
    arr = (arr // (256 // LEVELS)).astype(np.uint8)

    return arr


# ─────────────────────────────────────────────
# EKSTRAKSI FITUR GLCM
# ─────────────────────────────────────────────

def extract_features(gray: np.ndarray) -> dict:

    glcm = graycomatrix(
        gray,
        distances=DISTANCES,
        angles=ANGLES,
        levels=LEVELS,
        symmetric=True,
        normed=True
    )

    props = [
        "contrast",
        "homogeneity",
        "energy",
        "correlation"
    ]

    result = {}

    for p in props:
        val = graycoprops(glcm, p).mean()
        result[p] = float(round(val, 6))

    return result


# ─────────────────────────────────────────────
# PREDIKSI DECISION TREE
# ─────────────────────────────────────────────

def predict_class(feat: dict):

    feature_vector = np.array([
        feat["contrast"],
        feat["homogeneity"],
        feat["energy"],
        feat["correlation"]
    ]).reshape(1, -1)

    # Prediksi kelas
    pred = model.predict(feature_vector)[0]

    # Confidence probability
    prob = model.predict_proba(feature_vector)[0]

    kelas_list = model.classes_

    confidence = {
        kelas_list[i]: int(round(prob[i] * 100))
        for i in range(len(kelas_list))
    }

    # Biar total pas 100%
    total = sum(confidence.values())
    diff = 100 - total
    confidence[pred] += diff

    alasan_map = {
        "normal":
            "Permukaan logam terlihat homogen dan stabil.",

        "retak":
            "Terdeteksi pola garis tajam yang menyerupai retakan.",

        "korosi":
            "Terdeteksi pola permukaan tidak merata yang mengarah ke korosi."
    }

    return {
        "kelas": pred,
        "confidence": confidence,
        "alasan": alasan_map.get(pred, "-")
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({
            "error": "Tidak ada file gambar yang dikirim."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "Nama file kosong."
        }), 400

    img_bytes = file.read()

    try:
        gray = preprocess(img_bytes)

    except Exception as e:
        return jsonify({
            "error": f"Gagal memproses gambar: {str(e)}"
        }), 400

    # Ekstraksi fitur GLCM
    feat = extract_features(gray)

    # Prediksi pakai Decision Tree
    result = predict_class(feat)

    # Thumbnail preview
    thumb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    thumb.thumbnail((300, 300), Image.LANCZOS)

    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)

    thumb_b64 = base64.b64encode(
        buf.getvalue()
    ).decode()

    return jsonify({
        "kelas": result["kelas"],
        "confidence": result["confidence"],
        "alasan": result["alasan"],
        "glcm": feat,
        "thumbnail":
            f"data:image/jpeg;base64,{thumb_b64}",
        "filename": file.filename
    })


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Deteksi Cacat Permukaan Logam")
    print("  Metode: GLCM + Decision Tree")
    print("  http://127.0.0.1:5000")
    print("=" * 50 + "\n")

    app.run(
        debug=True,
        port=5000
    )