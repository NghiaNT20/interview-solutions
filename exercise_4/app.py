from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import joblib
import cv2

app = Flask(__name__)

svm_model = joblib.load("svm_model.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        file.save("upload.jpg")
        image = cv2.imread("upload.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = image.astype(float) / 255.0
        image = image.reshape(1, -1)
        result = svm_model.predict(image)
        return jsonify({"prediction": int(result[0])}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
