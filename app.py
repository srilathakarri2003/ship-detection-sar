"""
app.py

Flask web app for the Ship Detection from Satellite Images project.

Loads a trained CNN model and serves a simple web page where a user
can upload a satellite image patch and get a ship / no-ship prediction.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os

import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "model", "ship_detector.h5")
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMAGE_SIZE = 80

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the model once at startup rather than per-request
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(filepath: str):
    """Loads an image, preprocesses it, and returns (label, confidence)."""
    img = keras_image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Ship Detected" if prediction >= 0.5 else "No Ship Detected"
    confidence = float(prediction if prediction >= 0.5 else 1 - prediction)
    return label, confidence


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            result="Model not found. Train it first with train_model.py.",
        )

    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("index.html", result="No file selected.")

    if not allowed_file(file.filename):
        return render_template("index.html", result="Unsupported file type.")

    filename = secure_filename(file.filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    label, confidence = predict_image(filepath)
    image_url = f"/static/uploads/{filename}"

    return render_template(
        "index.html",
        result=label,
        confidence=f"{confidence * 100:.2f}%",
        image_url=image_url,
    )


if __name__ == "__main__":
    app.run(debug=True)
