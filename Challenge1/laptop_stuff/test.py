import torch
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Paths
image_path = 'gun_images/029.jpg'
npz_path = '29.npz'
text_prompt = "a photo of a gun"

# Load image
image = Image.open(image_path).convert("RGB")
original_image = np.array(image)

# Load local and global features from .npz file
data = np.load(npz_path)
local_features = data["features"]         # shape: (49, 768)
g_feature = data["g_feature"]             # shape: (768,)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Project 768D local features to 512D using CLIP visual projection
with torch.no_grad():
    local_features_tensor = torch.tensor(local_features, dtype=torch.float32).to(device)  # (49, 768)
    projected_features = local_features_tensor @ model.visual.proj  # (49, 512)
    projected_features = torch.nn.functional.normalize(projected_features, dim=1)  # Normalize

# Get text feature
with torch.no_grad():
    text_tokens = clip.tokenize([text_prompt]).to(device)
    text_features = model.encode_text(text_tokens)  # (1, 512)
    text_features = torch.nn.functional.normalize(text_features, dim=1)

# Compute cosine similarity (local)
similarities = projected_features @ text_features.T  # (49, 1)
similarity_map = similarities.squeeze().cpu().numpy()  # (49,)

# Reshape to spatial layout
patch_size = int(np.sqrt(similarity_map.shape[0]))  # should be 7
similarity_map = similarity_map.reshape(patch_size, patch_size)

# Normalize similarity map for visualization
similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-8)

# Resize to image size
similarity_map_resized = cv2.resize(similarity_map, (original_image.shape[1], original_image.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * similarity_map_resized), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Overlay
overlay = original_image * 0.6 + heatmap * 0.4
overlay = np.clip(overlay, 0, 255).astype(np.uint8)

# Global similarity computation

with torch.no_grad():
    g_feature_tensor = torch.tensor(g_feature, dtype=torch.float32).to(device)  # (512,)
    g_feature_tensor = torch.nn.functional.normalize(g_feature_tensor, dim=0)   # Ensure normalized
    global_similarity = (g_feature_tensor @ text_features.squeeze()).item()     # scalar


# Plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Similarity Map')
plt.imshow(similarity_map, cmap='jet')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f'Overlay\n"{text_prompt}"\nGlobal sim: {global_similarity:.3f}')
plt.imshow(overlay)
plt.axis('off')

plt.tight_layout()
plt.show()
