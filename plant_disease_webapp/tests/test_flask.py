"""
test_flask.py
=============
Automated tests for the Flask web application.

Run from the plant_disease_webapp/ directory:
    python -m pytest tests/ -v
or:
    python tests/test_flask.py
"""

import io
import os
import sys
import json

# Add parent dirs to path
TESTS_DIR  = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, WEBAPP_DIR)

import pytest
from app import app as flask_app

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.test_client() as c:
        yield c


def _real_image_bytes(cls="Tomato_Early_Blight"):
    """Load a real image from the parent project's data folder."""
    import glob
    pattern = os.path.join(WEBAPP_DIR, "..", "data", "plantvillage",
                           cls, "*.jpg")
    imgs = glob.glob(pattern)
    if not imgs:
        return None
    with open(imgs[0], "rb") as f:
        return f.read()


# ── Route tests ───────────────────────────────────────────────────────────

def test_homepage(client):
    """GET / should return 200 with HTML."""
    r = client.get("/")
    assert r.status_code == 200
    assert b"Plant Disease" in r.data


def test_about_page(client):
    """GET /about should return 200."""
    r = client.get("/about")
    assert r.status_code == 200


def test_health_endpoint(client):
    """GET /health should return JSON."""
    r    = client.get("/health")
    data = json.loads(r.data)
    assert "status" in data
    assert r.status_code in (200, 503)


# ── /predict error cases ──────────────────────────────────────────────────

def test_predict_no_file(client):
    """POST /predict with no file should return 400."""
    r    = client.post("/predict")
    data = json.loads(r.data)
    assert r.status_code == 400
    assert data["success"] is False


def test_predict_empty_filename(client):
    """POST /predict with empty filename should return 400."""
    r = client.post("/predict",
                    data={"file": (io.BytesIO(b""), "")},
                    content_type="multipart/form-data")
    data = json.loads(r.data)
    assert r.status_code == 400
    assert data["success"] is False


def test_predict_invalid_extension(client):
    """POST /predict with a .txt file should return 400."""
    r = client.post("/predict",
                    data={"file": (io.BytesIO(b"hello"), "test.txt")},
                    content_type="multipart/form-data")
    data = json.loads(r.data)
    assert r.status_code == 400
    assert data["success"] is False


def test_predict_corrupted_image(client):
    """POST /predict with a corrupt JPEG should return 500."""
    r = client.post("/predict",
                    data={"file": (io.BytesIO(b"not_an_image"), "bad.jpg")},
                    content_type="multipart/form-data")
    data = json.loads(r.data)
    assert data["success"] is False


# ── /predict success case ─────────────────────────────────────────────────

def test_predict_real_image(client):
    """POST /predict with a real leaf image should return a valid prediction."""
    img_bytes = _real_image_bytes("Tomato_Early_Blight")
    if img_bytes is None:
        pytest.skip("No real images found — run setup_real_data.py first.")

    r = client.post("/predict",
                    data={"file": (io.BytesIO(img_bytes), "leaf.jpg")},
                    content_type="multipart/form-data")
    data = json.loads(r.data)

    assert r.status_code == 200
    assert data["success"] is True
    assert "disease"     in data
    assert "display"     in data
    assert "confidence"  in data
    assert "all_probs"   in data
    assert "severity"    in data
    assert "action"      in data
    assert isinstance(data["confidence"], (int, float))
    assert 0 <= data["confidence"] <= 100
    assert len(data["all_probs"]) == 5


def test_predict_response_structure(client):
    """Verify all expected JSON keys are present in a successful response."""
    img_bytes = _real_image_bytes("Corn_Gray_Leaf_Spot")
    if img_bytes is None:
        pytest.skip("No real images found.")

    r    = client.post("/predict",
                       data={"file": (io.BytesIO(img_bytes), "corn.jpg")},
                       content_type="multipart/form-data")
    data = json.loads(r.data)

    required_keys = {"success", "disease", "display", "confidence",
                     "description", "severity", "action", "emoji", "all_probs"}
    assert required_keys.issubset(data.keys())


# ── Run directly ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=WEBAPP_DIR
    )
    sys.exit(result.returncode)
