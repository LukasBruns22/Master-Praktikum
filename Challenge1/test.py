import clip
import torch
from PIL import Image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device)  # You can change the model version if needed


# Load image and preprocess it
image_path = "messer/1.jpg"  # replace with your image file
image = Image.open(image_path)
image_input = preprocess(image).unsqueeze(0).to(device)

# Define your text input
text_input = ["a hand holding a knife", "a bird"]  # Add your own descriptions
text_input = clip.tokenize(text_input).to(device)

# Extract image features
with torch.no_grad():
    image_features = model.encode_image(image_input)

# Extract text features
with torch.no_grad():
    text_features = model.encode_text(text_input)

image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)


# Compute cosine similarity
similarity = (image_features @ text_features.T).squeeze(0)
print("Similarity between image and text:", similarity)
