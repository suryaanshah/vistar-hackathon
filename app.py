import streamlit as st
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import io # To handle image bytes

# --- Configuration ---
MODEL_LOAD_PATH = "pneumonia_classifier_model.pth" # Path to your saved model
CLASS_NAMES = ['NORMAL', 'PNEUMONIA'] # Must match the order from training

# --- Set up device ---
# We'll run inference on CPU for wider compatibility in a simple app
device = "cpu"
# Uncomment below if you want to try GPU (ensure CUDA is available)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device for inference.")


# --- Function to load the model ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model(model_path, num_classes):
    """Loads the pre-trained ResNet50 model with a modified classifier."""
    # Re-create the model architecture (must match training)
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Freeze parameters (optional but good practice)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(in_features=num_features, out_features=num_classes)

    # Load the saved state dictionary
    try:
        model.load_state_dict(torch.load(model_path, map_location=device)) # map_location ensures it loads to CPU/GPU correctly
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Make sure it's in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"Error loading model state: {e}")
        return None

    model = model.to(device)
    model.eval() # Set model to evaluation mode
    return model

# --- Define Image Transformations ---
# IMPORTANT: Must match the validation/test transforms used during training
infer_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Function to make prediction ---
def predict(model, image_bytes):
    """Takes image bytes, preprocesses, and returns prediction."""
    try:
        # Load image using PIL
        img = Image.open(io.BytesIO(image_bytes))
        # Convert grayscale images to RGB (X-rays are often grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Apply transformations
        img_tensor = infer_transforms(img).unsqueeze(0) # Add batch dimension
        img_tensor = img_tensor.to(device)

        # Make prediction
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = CLASS_NAMES[prediction_idx]
            confidence = probabilities[0][prediction_idx].item()

        return predicted_class, confidence

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# --- Streamlit App Layout ---

st.title("🩺 Pneumonia Detector from Chest X-Rays")
st.write("Upload a chest X-ray image, and the AI will predict if it shows signs of pneumonia.")

# Load the model (cached)
model = load_model(MODEL_LOAD_PATH, len(CLASS_NAMES))

if model is not None:
    # File uploader
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes, caption='Uploaded X-Ray', use_container_width=True)

        # Make prediction when button is clicked
        if st.button('Classify Image'):
            with st.spinner('Analyzing...'):
                predicted_class, confidence = predict(model, image_bytes)

            if predicted_class is not None:
                st.success(f"Prediction: **{predicted_class}**")
                st.info(f"Confidence: {confidence:.2f}")
            else:
                st.error("Could not make a prediction. Please check the image or model.")
else:
    st.warning("Model could not be loaded. Please ensure the model file is present.")

st.sidebar.info(
    "**Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical diagnosis."
)