# Plant Disease Detection — Flask Web Application

A mobile-friendly web application that lets farmers upload leaf photos and receive instant plant disease predictions using a classical machine learning ensemble.

---

## Features

- Upload leaf photos from phone or computer (drag-and-drop + click)
- Instant disease detection using KNN + SVM + Random Forest ensemble
- Confidence percentage and probability breakdown per class
- Disease description, severity rating, and recommended action
- Mobile-first responsive design — works on any screen size
- REST API endpoint for integration with mobile apps
- No deep learning — lightweight and deployable on low-end servers

---

## Supported Diseases

| Class | Crop | Type |
|-------|------|------|
| Tomato Early Blight | Tomato | Disease |
| Tomato Late Blight | Tomato | Disease |
| Tomato Healthy | Tomato | Healthy |
| Corn Gray Leaf Spot | Corn | Disease |
| Potato Early Blight | Potato | Disease |

---

## Installation

```bash
# 1. Navigate to this folder
cd plant_disease_webapp

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** This app reuses the trained models from the parent project (`../models/`).
> Run `python ../run_pipeline.py` first if the models do not exist yet.

---

## Run Locally

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

To enable debug mode:
```bash
set FLASK_DEBUG=true   # Windows
python app.py
```

---

## Run with Gunicorn (Production)

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## Deploy to Render (Free Tier)

1. Push this folder to a GitHub repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn -w 2 -b 0.0.0.0:$PORT app:app`
5. Add environment variables if needed (see below)
6. Deploy

> **Important:** Add your trained `.pkl` model files to `../models/` before deploying,
> or set `MODEL_PATH` environment variable to point to your models directory.

---

## Deploy to Railway

1. Push to GitHub
2. New project → Deploy from GitHub repo
3. Railway auto-detects Python and installs `requirements.txt`
4. Set start command: `gunicorn -w 2 -b 0.0.0.0:$PORT app:app`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `../models` | Path to the directory containing `.pkl` model files |
| `PORT` | `5000` | Port to run the server on |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |

---

## API Documentation

### `POST /predict`

Accepts a leaf image and returns a disease prediction.

**Request:**
```
Content-Type: multipart/form-data
Field: file  (image/jpeg or image/png, max 16MB)
```

**Response (success):**
```json
{
  "success"    : true,
  "disease"    : "Tomato_Early_Blight",
  "display"    : "Tomato Early Blight",
  "confidence" : 87.3,
  "severity"   : "Moderate",
  "description": "Caused by Alternaria solani. Brown spots with concentric rings.",
  "action"     : "Remove infected leaves. Apply copper-based fungicide.",
  "emoji"      : "🍅",
  "all_probs"  : {
    "Tomato Early Blight" : 87.3,
    "Tomato Late Blight"  : 6.1,
    "Tomato Healthy"      : 3.5,
    "Corn Gray Leaf Spot" : 2.2,
    "Potato Early Blight" : 0.9
  }
}
```

**Response (error):**
```json
{
  "success": false,
  "error"  : "Human-readable error message"
}
```

**HTTP status codes:**
- `200` — successful prediction
- `400` — bad request (no file, invalid type)
- `429` — rate limit exceeded (10 per minute per IP)
- `500` — prediction or server error
- `503` — models not loaded

### `GET /health`
Returns `{"status": "ok", "models_loaded": true}` when ready.

---

## Model Training

Models are trained by the parent project pipeline:

```bash
cd ..
python run_pipeline.py
```

This trains KNN, SVM, and Random Forest classifiers on PlantVillage + PlantDoc images
and saves `.pkl` files to `../models/`. The Flask app loads them automatically.

---

## Project Structure

```
plant_disease_webapp/
├── app.py                  ← Flask server
├── requirements.txt
├── README.md
├── .gitignore
├── robots.txt
├── src/
│   └── __init__.py         ← path shim to parent src/
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/            ← temporary upload storage
├── templates/
│   ├── index.html          ← main detection page
│   └── about.html          ← project info page
└── tests/
    └── test_flask.py       ← automated tests
```

---

## Dataset Information

Models were trained on a **subset** of two public datasets:

| Dataset | Images in project | Original size |
|---------|-------------------|---------------|
| PlantVillage (lab) | 5,000 | ~54,306 |
| PlantDoc (field) | 447 | ~2,598 |
| **Total** | **5,447** | — |

---

## Limitations

- Works best on single-leaf close-up photos
- Only classifies the 5 supported disease classes
- Classical ML — not a deep learning model
- Field (PlantDoc) accuracy ~42% vs 92.5% on lab images

---

## License

For academic use. Dataset licenses: PlantVillage (CC BY 4.0), PlantDoc (CC BY 4.0).
