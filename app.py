import os
import torch
import torch.nn as nn
import pandas as pd
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# Load CSV data
# --------------------------------------------------
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# --------------------------------------------------
# Download model from HuggingFace (handles LFS)
# --------------------------------------------------
print("Downloading model from HuggingFace Hub...")

MODEL_PATH = hf_hub_download(
    repo_id="yashj3238/green-model",
    filename="ResNet50.pt"
)

print("Model downloaded at:", MODEL_PATH)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = resnet50(weights=None)
num_classes = 39
model.fc = nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully!")

# --------------------------------------------------
# Image preprocessing (IMPORTANT for ResNet)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict(file_path):
    try:
        img = Image.open(file_path).convert("RGB")
        img = transform(img)

        with torch.no_grad():
            output = model(img.unsqueeze(0))
            predicted_class = torch.argmax(output)

        return predicted_class.item()

    except Exception as e:
        print("Prediction error:", e)
        return None

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

# --------------------------------------------------
# Submit Route
# --------------------------------------------------
@app.route('/submit', methods=['POST'])
def submit():
    image = request.files['image']
    filename = image.filename

    upload_dir = os.path.join('static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)
    image.save(file_path)

    pred = predict(file_path)

    if pred is None:
        return "Prediction failed. Try another image.", 500

    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]

    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    return render_template(
        'submit.html',
        title=title,
        desc=description,
        prevent=prevent,
        image_url=image_url,
        pred=pred,
        sname=supplement_name,
        simage=supplement_image_url,
        buy_link=supplement_buy_link
    )

# --------------------------------------------------
# Market Route
# --------------------------------------------------
@app.route('/market')
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

# --------------------------------------------------
# Run (for local only — Render uses gunicorn)
# --------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
