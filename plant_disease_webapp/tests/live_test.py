"""Live integration test — Flask server must be running on port 5000."""
import sys, os, glob, json
import requests

BASE = "http://localhost:5000"

def test_live(cls, img_path):
    with open(img_path, "rb") as f:
        r = requests.post(f"{BASE}/predict",
                          files={"file": (os.path.basename(img_path), f, "image/jpeg")})
    d = r.json()
    status = "PASS" if (r.status_code == 200 and d.get("success")) else "FAIL"
    print(f"  [{status}] {cls:<30s} → {d.get('display','?'):<30s}  conf={d.get('confidence','?')}%")
    return d.get("success", False)

if __name__ == "__main__":
    ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "plantvillage")
    classes = ["Tomato_Early_Blight","Tomato_Late_Blight","Tomato_Healthy",
               "Corn_Gray_Leaf_Spot","Potato_Early_Blight"]
    print(f"\nLive prediction test against {BASE}")
    print("-" * 70)
    passed = 0
    for cls in classes:
        imgs = glob.glob(os.path.join(ROOT, cls, "*.jpg"))
        if not imgs:
            print(f"  [SKIP] {cls} — no images")
            continue
        if test_live(cls, imgs[2]):
            passed += 1
    print(f"\n{passed}/{len(classes)} passed")
