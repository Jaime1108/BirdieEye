import logging
import os
from typing import List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

######## Models ##########
birdValid = ""
bird_detection_model = "baseline_cnn_bird_detector.keras"
CNNModel = "baseline_cnn.keras"

ResNetModel = "resnet50_finetuned.keras"
EfficientNetModel = "efficientnet_finetuned.keras"
modelName = CNNModel    
##############
############################
with open("class_names.json", "r") as f:
    class_names = json.load(f)
############################


print("Using model path:", modelName)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
TARGET_SIZE: Tuple[int, int] = (224, 224)

isBird = [True, False]

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


with open("class_names.json", "r") as f:
    class_names = json.load(f)

def load_labels(path: str = "labels.txt") -> List[str]:
    """Return class labels if a labels.txt file exists."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


labels: List[str] = load_labels()




def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_stream) -> np.ndarray:
    """Prepare the uploaded image for the model."""
    image = Image.open(file_stream).convert("RGB")
    image = image.resize(TARGET_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_species(file_stream, model) -> Optional[dict]:
    if model is None:
        return None

    image_tensor = preprocess_image(file_stream)
    prediction = model.predict(image_tensor)
    top_index = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))
    species_name = labels[top_index] if labels and top_index < len(labels) else f"class_{top_index}"
    return {
        "species": decipherSpeciesName(species_name),
        "confidence": round(confidence, 4),
        "raw_prediction": prediction[0].tolist(),
    }
def predict_valid_bird(file_stream) -> bool:
    if bird_detection_model is None:
        return None

    image_tensor = preprocess_image(file_stream)
    prediction = tf.keras.models.load_model(bird_detection_model).predict(image_tensor)
    top_index = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))
    choice = labels[top_index] if labels and top_index < len(labels) else f"class_{top_index}"
    return decipherIsBird(choice)
def decipherSpeciesName(output):
    print("Raw output:", output)
    birdNameIndex = decipherPrediction(output)
    print("Bird Name Index:", birdNameIndex)
    birdName = class_names[int(birdNameIndex)]
    birdNameClean = birdName.replace("_", " ")
    return birdNameClean

def decipherPrediction(output):
    print("Raw output:", output)
    predictionIndex = output[6:]
    print("Bird Name Index:", predictionIndex)
    return int(predictionIndex)

def decipherIsBird(output)  -> bool:
    print("Raw output:", output)
    choiceIndex = decipherPrediction(output)
    print("choice Index:", choiceIndex)
    return isBird[int(choiceIndex)]


def choosingModel(option):
    match option:
        case "baselineCNN":
            return CNNModel
        case "resnet50":
            return ResNetModel
        case _:
            return EfficientNetModel
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    isValidBird = True
    if request.method == "POST":
        print("Model selected:", request.form.get("model"))
        modelName = choosingModel(request.form.get("model"))
        file = request.files.get("image")
        ############
        '''
        print("check for valid bird images")
        isValidBird = predict_valid_bird(file.stream)
        print("isValidBird:", isValidBird)
        if isValidBird is False:
            print("Invalid bird image")
            error = "The image you upload is not a valid bird image."
            return render_template("index.html", prediction=None, error=error)
        '''
        #############    
        try:
            logger.info("Loading model from %s", modelName)
            model = tf.keras.models.load_model(modelName)
            model.summary()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to load model: %s", exc)
            model = None

        if not file or file.filename == "":
            error = "Please choose an image to upload."
        elif not allowed_file(file.filename):
            error = "Unsupported file type. Please upload a JPG or PNG image."
        else:
            filename = secure_filename(file.filename)
            logger.info("Loading model from %s", model)
            logger.info("Received upload: %s", filename)
            try:
                file.stream.seek(0)
                prediction = predict_species(file.stream, model)
                if prediction is None:
                    error = "Model failed to predict."
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Prediction failed: %s", exc)
                error = "Could not process the image. Please try a different file."

    return render_template("index.html", prediction=prediction, error=error)


@app.route("/predict", methods=["POST"])
def predict_api():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a JPG or PNG image."}), 400
    try:
        file.stream.seek(0)
        prediction = predict_species(file.stream)
        if prediction is None:
            return jsonify({"error": "Model failed to load. Check that test-agent.h5 is available."}), 500
        return jsonify(prediction)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("API prediction failed: %s", exc)
        return jsonify({"error": "Could not process the image."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
