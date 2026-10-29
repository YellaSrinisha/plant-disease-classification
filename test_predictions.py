"""Quick test — run predictions on real images from each class."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from src.predict import load_all_models, predict_image

models = load_all_models()

classes = [
    'Tomato_Early_Blight',
    'Tomato_Late_Blight',
    'Tomato_Healthy',
    'Corn_Gray_Leaf_Spot',
    'Potato_Early_Blight',
]

print("\nTesting on PlantVillage (lab) images:")
print("-" * 55)
for cls in classes:
    imgs = sorted(glob.glob(f'data/plantvillage/{cls}/*.jpg'))[:5]
    results = []
    for p in imgs:
        pil = Image.open(p)
        r   = predict_image(pil, models)
        pred = r['predictions']['ensemble']['label']
        results.append("OK" if pred == cls else f"WRONG({pred})")
    print(f"  {cls:<30s}: {' | '.join(results)}")

print("\nTesting on PlantDoc (field) images:")
print("-" * 55)
for cls in classes:
    imgs = sorted(glob.glob(f'data/plantdoc/{cls}/*.jpg'))[:5]
    if not imgs:
        print(f"  {cls:<30s}: no images")
        continue
    results = []
    for p in imgs:
        pil = Image.open(p)
        r   = predict_image(pil, models)
        pred = r['predictions']['ensemble']['label']
        results.append("OK" if pred == cls else f"WRONG({pred})")
    print(f"  {cls:<30s}: {' | '.join(results)}")
