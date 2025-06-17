import os
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from dotenv import load_dotenv

load_dotenv()
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def describe_image(path):
    image = Image.open(path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def extract_parameters(caption):
    return {
        "crosswalk": "crosswalk" in caption,
        "light": "light" in caption or "signal" in caption,
        "pothole": "pothole" in caption or "damaged" in caption,
        "lighting": "well lit" in caption or "daylight" in caption
    }

results = []
for fname in sorted(os.listdir(IMG_DIR)):
    if fname.endswith(".jpg"):
        path = os.path.join(IMG_DIR, fname)
        caption = describe_image(path)
        params = extract_parameters(caption)
        results.append({"image": fname, "caption": caption, **params})

df = pd.DataFrame(results)
os.makedirs("data", exist_ok=True)
df.to_csv("data/safety_parameters.csv", index=False)
print("Done. CSV saved to data/safety_parameters.csv")
