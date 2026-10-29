"""
verify_routes.py
Verifies all Flask routes are working correctly against live server.
Run: python tests/verify_routes.py
"""
import sys, os, glob, json, io
import requests

BASE = "http://localhost:5000"
PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

def check(name, cond, detail=""):
    sym = PASS if cond else FAIL
    print(f"  {sym}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        errors.append(name)

print(f"\nVerifying Flask routes at {BASE}")
print("=" * 55)

# GET /
r = requests.get(f"{BASE}/")
check("GET /  returns 200",      r.status_code == 200)
check("GET /  contains 'Plant'", "Plant" in r.text)

# GET /about
r = requests.get(f"{BASE}/about")
check("GET /about returns 200",         r.status_code == 200)
check("GET /about contains 'Overview'", "Overview" in r.text or "Project" in r.text)

# GET /results
r = requests.get(f"{BASE}/results")
check("GET /results returns 200",       r.status_code == 200)
check("GET /results contains metrics",  "KNN" in r.text or "Evaluation" in r.text)
check("GET /results contains accuracy", "Accuracy" in r.text or "accuracy" in r.text)

# GET /health
r = requests.get(f"{BASE}/health")
d = r.json()
check("GET /health returns 200",          r.status_code == 200)
check("GET /health models_loaded = true", d.get("models_loaded") is True)

# POST /predict — real image
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "plantvillage")
imgs = glob.glob(os.path.join(ROOT, "Tomato_Early_Blight", "*.jpg"))
if imgs:
    with open(imgs[5], "rb") as f:
        r = requests.post(f"{BASE}/predict",
                          files={"file": ("leaf.jpg", f, "image/jpeg")})
    d = r.json()
    check("POST /predict returns 200",        r.status_code == 200)
    check("POST /predict success=true",       d.get("success") is True)
    check("POST /predict has disease",        "disease" in d)
    check("POST /predict has confidence",     "confidence" in d)
    check("POST /predict has all_probs (5)",  len(d.get("all_probs", {})) == 5)
    check("POST /predict has severity",       "severity" in d)
    check("POST /predict has action",         "action" in d)
    print(f"\n         Predicted: {d.get('display')}  ({d.get('confidence')}%)")
else:
    print("  [SKIP] No real images found for /predict test")

# GET /report-figure
r = requests.get(f"{BASE}/report-figure/model_comparison_accuracy.png")
check("GET /report-figure serves PNG",  r.status_code == 200)
check("GET /report-figure is PNG data", r.headers.get("Content-Type","").startswith("image"))

print()
print("=" * 55)
if not errors:
    print(f"  All checks passed.")
else:
    print(f"  {len(errors)} check(s) FAILED: {errors}")

sys.exit(0 if not errors else 1)
