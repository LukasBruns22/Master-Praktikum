import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from torchvision.transforms.functional import resize
from torchvision import transforms

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model.eval()

# Function to resize numpy array using torch
def resize_array(array, size):
    """Resize a numpy array using PyTorch's resize function"""
    # Convert to tensor, add batch and channel dimensions
    tensor = torch.tensor(array).float().unsqueeze(0).unsqueeze(0)
    # Resize
    resized = transforms.functional.resize(tensor, size, antialias=True)
    # Convert back to numpy and remove extra dimensions
    return resized.squeeze().numpy()

def get_text_patch_similarities(image_path, text_prompts):
    """
    Get similarities between image patches and text prompts.
    Returns the original image and a list of similarity maps.
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device).half()
    
    # Process image to get patch features
    with torch.no_grad():
        # Run through the first convolutional layer
        x = model.visual.conv1(image_tensor)  # [1, width, grid, grid]
        grid_size = x.shape[-1]  # Should be 24 for ViT-L/14@336px
        
        # Convert to sequence of patches
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [1, N, width]
        
        # Add class token
        class_embedding = model.visual.class_embedding.unsqueeze(0).unsqueeze(0).to(x.dtype)
        class_embedding = class_embedding.expand(x.shape[0], 1, -1)
        x = torch.cat([class_embedding, x], dim=1)  # [1, N+1, width]
        
        # Add positional embeddings
        x = x + model.visual.positional_embedding.to(x.dtype)[:x.shape[1]]
        x = model.visual.ln_pre(x)
        
        # Apply transformer
        x = x.permute(1, 0, 2)  # [N+1, 1, width]
        x = model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # [1, N+1, width]
        
       # Extract patch features BEFORE projection
        similarity_maps = []
        patch_features = x[:, 1:, :]  # skip CLS token
        patch_features = patch_features @ model.visual.proj  # [1, N, 512]
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

        # Now compare to text embeddings
        for prompt in text_prompts:
            text_tokens = clip.tokenize([prompt]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarities = (100 * patch_features @ text_features.T).squeeze()
            similarity_map = similarities.reshape(grid_size, grid_size).cpu().numpy()
            similarity_maps.append(similarity_map)
            
            return image, similarity_maps

def create_heatmap_overlay(image, similarity_map, alpha=0.6):
    """Create a heatmap overlay on the image without using cv2"""
    # Resize similarity map to match image dimensions
    h, w = image.height, image.width
    resized_map = resize_array(similarity_map, (h, w))
    
    # Create a figure
    plt.figure(figsize=(10, 10))
    
    # Display the original image
    plt.imshow(image)
    
    # Overlay the heatmap
    plt.imshow(resized_map, cmap='jet', alpha=alpha, interpolation='bilinear')
    
    plt.axis('off')
    
    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    # Convert canvas to an array
    overlay_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    
    return overlay_img

def visualize_similarities(image_path, text_prompts):
    """Visualize similarities between image patches and text prompts"""
    # Get similarities
    image, similarity_maps = get_text_patch_similarities(image_path, text_prompts)
    
    # Create figure
    fig, axes = plt.subplots(1, len(text_prompts), figsize=(18, 6))
    if len(text_prompts) == 1:
        axes = [axes]
    
    # For each prompt
    for i, (prompt, sim_map) in enumerate(zip(text_prompts, similarity_maps)):
        # Normalize the map for better visualization
        vmin = sim_map.min()
        vmax = sim_map.max()
        
        # Resize similarity map to match image dimensions
        h, w = image.height, image.width
        resized_map = resize_array(sim_map, (h, w))
        
        # Display original image
        axes[i].imshow(image)
        
        # Overlay heatmap
        im = axes[i].imshow(resized_map, cmap='jet', alpha=0.6, 
                           interpolation='bicubic', vmin=vmin, vmax=vmax)
        
        axes[i].set_title(f"'{prompt}'")
        axes[i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('clip_patch_text_similarities.png', dpi=300)
    plt.show()

# Main execution
image_path = "Messer/1.jpg"
text_prompts = ["a hand", "a knife", "random text aksdjfh"]

# Generate and visualize patch-text similarities
visualize_similarities(image_path, text_prompts)

"""
import torch
import clip
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Load CLIP model
model, preprocess = clip.load("ViT-L/14@336px", device="cuda")
model.eval()

# Load and preprocess image
original_image = Image.open("Messer/1.jpg").convert("RGB")
image = preprocess(original_image).unsqueeze(0).to("cuda").half()

# Encode text
text = clip.tokenize(["abcsdjkfls"]).to("cuda")
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

with torch.no_grad():
    x = model.visual.conv1(image)               # [1, 768, 7, 7]
    x = x.reshape(x.shape[0], x.shape[1], -1)   # [1, 768, 49]
    x = x.permute(0, 2, 1)                      # [1, 49, 768]
    class_emb = model.visual.class_embedding.to(x.dtype)
    class_emb = class_emb.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
    x = torch.cat([class_emb, x], dim=1)        # [1, 50, 768]
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)
    x = x.permute(1, 0, 2)                      # [50, 1, 768]
    x = model.visual.transformer(x)
    x = x.permute(1, 0, 2)                      # [1, 50, 768]
    patch_features = x[:, 1:, :]                # discard CLS token -> [1, 49, 768]
    #patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
    patch_features = patch_features @ model.visual.proj  # ✅ matrix multiplication
    patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

# Compute similarity heatmap
similarity = (patch_features @ text_features.T).squeeze()  # [N] or [1, N]
#print("Similarity between image and text:", similarity)
grid_size = int(model.visual.positional_embedding.shape[0]**0.5)  # exclude CLS
heatmap = similarity.reshape(grid_size, grid_size).cpu().numpy()
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())


# Upsample heatmap
heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
heatmap_upsampled = F.interpolate(heatmap_tensor, size=(original_image.size[1], original_image.size[0]), mode='bilinear', align_corners=False)
heatmap_upsampled = heatmap_upsampled.squeeze().numpy()

# Overlay on image
plt.imshow(original_image)
plt.imshow(heatmap_upsampled, cmap='jet', alpha=0.5)  # alpha controls transparency
plt.axis('off')
plt.colorbar(label="Similarity")
plt.title("CLIP Heatmap Overlay")
plt.show()
"""