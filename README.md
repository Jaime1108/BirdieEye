# BirdieEye

Simple Flask service that loads `test-agent.h5` and predicts the bird species from an uploaded image. Includes an HTML form at `/` and a JSON API at `/predict`.

## Setup
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your trained Keras model at `test-agent.h5` in this folder. Optionally add a `labels.txt` file (one class label per line) to map class indices to species names.

## Run the server
```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```
Then open http://localhost:5000 to use the upload form.

## API
`POST /predict` with `form-data` key `image` containing a JPG/PNG file.

Response (example):
```json
{
  "species": "northern_cardinal",
  "confidence": 0.93,
  "raw_prediction": [0.01, 0.93, 0.06]
}
```

If the model cannot be loaded, the service returns an error indicating the missing `test-agent.h5`.
