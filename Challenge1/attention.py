import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Example: Load your image and text
image_path = "Messer/1.jpg"  # Replace with the path to your image
text = "a photo of a knife being held by a hand"  # Replace with your prompt

# Load the image
from PIL import Image
image = Image.open(image_path)

# Preprocess the image and text
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

# Forward pass through the model and set it to return attention weights
outputs = model(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'], output_attentions=True)

# Extract attention maps from the model's outputs
attentions = outputs.attentions  # List of attention weights for each layer

# Example: Visualize the attention map from the first layer and first attention head
attention_map = attentions[0][0, 0, :, :].detach().cpu().numpy()  # Attention map shape: [num_tokens, num_tokens]

# Resize the attention map to match the image size
image_size = (224, 224)  # Resize to match the input image size
attention_map_resized = np.resize(attention_map, image_size)

# Normalize the attention map for better visualization
attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())

# Plot the image and overlay the attention map
fig, ax = plt.subplots()
ax.imshow(image)  # Plot the original image
ax.imshow(attention_map_resized, cmap="hot", alpha=0.5)  # Overlay the attention map with transparency
plt.title("Attention Map (First Layer, First Head)")
plt.show()

# Optional: Visualize attention maps from all layers and attention heads
for layer in range(len(attentions)):
    for head in range(attentions[layer].shape[1]):  # Iterate through attention heads
        attention_map = attentions[layer][0, head, :, :].detach().cpu().numpy()
        
        # Resize and normalize the attention map
        attention_map_resized = np.resize(attention_map, image_size)
        attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
        
        # Plot the attention map
        plt.imshow(image)  # Plot the original image
        plt.imshow(attention_map_resized, cmap="hot", alpha=0.5)  # Overlay the heatmap
        plt.title(f"Attention Map - Layer {layer+1}, Head {head+1}")
        plt.show()
