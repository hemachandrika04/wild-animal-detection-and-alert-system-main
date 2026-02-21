import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import vgg19
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from twilio.rest import Client
import os
import config # Import your configuration file

# --- Page Configuration ---
st.set_page_config(
    page_title="Wildlife Detection & Alert System",
    page_icon="🐾",
    layout="wide"
)

# --- 1. Configuration & Constants ---
MODEL_PATH = 'artifacts/wild_model.pth'
CLASS_NAMES = ['bear', 'chinkara', 'elephant', 'lion', 'peacock', 'pig', 'sheep', 'tiger']
DANGEROUS_ANIMALS = ['bear', 'elephant', 'lion', 'tiger', 'pig']
CONFIDENCE_THRESHOLD = 0.80 # Set a threshold for sending alerts

# --- 2. Model Definition ---
# This class must be the same as the one used during training.
class VGG19_Classifier(nn.Module):
    def __init__(self, num_classes=8):
        super(VGG19_Classifier, self).__init__()
        # Load a pretrained VGG19 model
        self.vgg = vgg19(pretrained=True)
        # Freeze feature extraction layers
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        # Replace the classifier with the one matching your trained model
        in_features = self.vgg.classifier[0].in_features
        self.vgg.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.vgg(x)

# --- 3. Model Loading & Caching ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model(model_path, num_classes):
    """Loads the PyTorch model from a .pth file."""
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG19_Classifier(num_classes=num_classes)
    
    # Load the state_dict directly into the model's `vgg` attribute
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.vgg.load_state_dict(state_dict)
    except RuntimeError as e:
        st.error(f"Error loading model weights. Ensure the model architecture in the script matches the one used for training. Details: {e}")
        st.stop()
        
    model.to(device)
    model.eval()
    return model, device

# Load the model and device
model, device = load_model(MODEL_PATH, len(CLASS_NAMES))

# --- 4. Image Transformations ---
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# --- 5. Core Functions ---
def send_alert(animal_name):
    """Sends an SMS alert using Twilio credentials from config.py."""
    try:
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        message_body = (
            f"🚨 WILDLIFE ALERT! 🚨\n"
            f"A dangerous animal has been detected.\n"
            f"Animal: {animal_name.capitalize()}"
        )
        message = client.messages.create(
            body=message_body,
            from_=config.TWILIO_PHONE_NUMBER,
            to=config.RECIPIENT_PHONE_NUMBER
        )
        st.success(f"✅ Alert message sent successfully! SID: {message.sid}")
    except Exception as e:
        st.error(f"❌ Failed to send Twilio alert: {e}")

def predict_image(model, image, device):
    """Preprocesses an image and returns the prediction and confidence."""
    image_np = np.array(image)
    transformed = val_transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class_name = CLASS_NAMES[predicted_idx.item()]
    return predicted_class_name, confidence.item()

# --- 6. Streamlit User Interface ---
st.title("🐾 Wildlife Detection and Alert System")
st.markdown("Upload an image of an animal, and the system will identify it. If a dangerous animal is detected with high confidence, an alert will be sent.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
    with col2:
        with st.spinner('Analyzing the image...'):
            # Make a prediction
            predicted_animal, confidence = predict_image(model, image, device)
            
            # Display the prediction
            st.subheader("Prediction Result")
            st.markdown(f"**Detected Animal:** `{predicted_animal.capitalize()}`")
            st.markdown(f"**Confidence:** `{confidence:.2%}`")
            
            # Progress bar for confidence
            st.progress(confidence)
            
            # Check if an alert should be sent
            st.subheader("Alert Status")
            is_dangerous = predicted_animal.lower() in DANGEROUS_ANIMALS
            is_confident = confidence > CONFIDENCE_THRESHOLD

            if is_dangerous and is_confident:
                st.warning(f"⚠️ **Dangerous Animal Detected!** Sending an alert...")
                send_alert(predicted_animal)
            elif is_dangerous and not is_confident:
                st.info(f"'{predicted_animal.capitalize()}' is a dangerous animal, but confidence is below the {CONFIDENCE_THRESHOLD:.0%} threshold. No alert sent.")
            else:
                st.success("✅ Animal is not classified as dangerous. No alert sent.")