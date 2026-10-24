import os
import socket

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from keras.models import load_model
from keras.utils import normalize
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_SIZE = 64  # training me bhi 64 tha
MODEL_PATH = "BrainTumor10EpochsCategorical.h5"  # apna trained model isi folder me rakho
ALLOWED_EXT = (".jpg", ".jpeg", ".png")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"'{MODEL_PATH}' nahi mila. Model file ko app.py ke saath same folder me rakho."
    )

model = load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)


# ---------------------------------------------------------------------------
# Preprocessing (bilkul training jaisa)
# ---------------------------------------------------------------------------
def preprocess(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if image is None:
        raise ValueError("Image read nahi ho paayi.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((INPUT_SIZE, INPUT_SIZE))

    x = np.array(image)
    x = np.expand_dims(x, axis=0)  # (1, 64, 64, 3)
    x = normalize(x, axis=1)  # training me bhi axis=1 use hua tha
    return x


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify(error="Koi file upload nahi hui."), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify(error="Koi file select nahi ki."), 400
    if not file.filename.lower().endswith(ALLOWED_EXT):
        return jsonify(error="Sirf JPG, JPEG ya PNG image allowed hai."), 400

    try:
        x = preprocess(file.read())
        probs = model.predict(x, verbose=0)[0]  # [P(no tumor), P(tumor)]
    except Exception as e:
        return jsonify(error=f"Prediction fail hui: {e}"), 500

    no_tumor = float(probs[0])
    tumor = float(probs[1])
    has_tumor = tumor > no_tumor

    return jsonify(
        has_tumor=has_tumor,
        label="Tumor Detected" if has_tumor else "No Tumor Detected",
        confidence=round(max(tumor, no_tumor) * 100, 2),
        tumor_probability=round(tumor * 100, 2),
        no_tumor_probability=round(no_tumor * 100, 2),
    )


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error="File bahut badi hai (max 10 MB)."), 413


def get_lan_ip() -> str:
    """Laptop ka local network (WiFi) IP nikalta hai, mobile se kholne ke liye."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # koi data nahi bhejta, bas sahi network interface pata karta hai
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))  # server apna PORT khud deta hai
    ip = get_lan_ip()
    print("\n" + "=" * 55)
    print("  NeuroScan AI chal raha hai!")
    print(f"  Laptop par :  http://127.0.0.1:{PORT}")
    print(f"  Mobile par :  http://{ip}:{PORT}   (same WiFi zaroori)")
    print("=" * 55 + "\n")

    # host="0.0.0.0" se laptop aur mobile dono se access hoga
    app.run(host="0.0.0.0", port=PORT, debug=False)
