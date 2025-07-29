import torch
import torch.nn.functional as F
from flask import Flask, request, render_template
from PIL import Image
import os

# Import your project's modules
from src import config
from src import model as model_def
from src import dataset as dataset_def

# --- Configuration ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load The Model (Do this only once) ---
MODEL_NAME = config.TRAINING_CONFIG['model_name']
# PASTE THE FULL, CORRECT PATH TO YOUR BEST SAVED MODEL
MODEL_PATH = "Alzheimer_Dataset/saved_models/EfficientNet_FocalLoss_Mixup_efficientnet_b3_2025-07-28_21-56-54.pth" 

model = model_def.create_model(MODEL_NAME)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
model.to(config.DEVICE)
model.eval()
print(f"Model {MODEL_NAME} loaded successfully from {MODEL_PATH}")

CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
transforms = dataset_def.get_val_test_transforms()


def get_prediction(image_path):
    """Takes an image path and returns the prediction."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transforms(image).unsqueeze(0)
        image_tensor = image_tensor.to(config.DEVICE)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item() * 100
        
        return f"{predicted_class} (Confidence: {confidence_score:.2f}%)"
    except Exception as e:
        return f"Error processing image: {e}"


@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in the request.")
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            return render_template('index.html', error="No file selected.")
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Get prediction
            prediction = get_prediction(filepath)
            
            return render_template('index.html', prediction_text=prediction)
            
    # This is for the initial page load (GET request)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)