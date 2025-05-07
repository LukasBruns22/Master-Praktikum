import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from torchvision.ops import nms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load Faster R-CNN for RPN
rpn_model = fasterrcnn_resnet50_fpn(pretrained=True).eval()

# Load image
image_path = image_path = r"A:\studium\München\Praktikum\repo\Master-Praktikum\Challenge1\gun_images\035.jpg"
  # Replace with your image
image_pil = Image.open(image_path).convert("RGB")
image_tensor = ToTensor()(image_pil)

# Get image size
W, H = image_pil.size
heatmap = torch.zeros((H, W))
count_map = torch.zeros((H, W))

# Get proposals
with torch.no_grad():
    outputs = rpn_model([image_tensor])[0]

boxes = outputs["boxes"]
scores = outputs["scores"]

# Filter boxes and scores based on the score threshold
score_thresh = 0.5
keep = scores > score_thresh
boxes = boxes[keep]
scores = scores[keep]

# Apply Non-Maximum Suppression (NMS)
nms_indices = nms(boxes, scores, iou_threshold=0.5)

# Now apply the nms indices to get the final boxes
boxes = boxes[nms_indices]

# Encode text
prompt = "a gun"  # change as needed
text_inputs = clip_processor(text=[prompt], return_tensors="pt")
with torch.no_grad():
    text_features = clip_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Accumulate heatmap
for box in boxes:
    x1, y1, x2, y2 = map(int, box.tolist())
    if x2 - x1 < 10 or y2 - y1 < 10:  # skip tiny boxes
        continue

    region = image_pil.crop((x1, y1, x2, y2))
    inputs = clip_processor(images=region, return_tensors="pt")
    with torch.no_grad():
        img_feat = clip_model.get_image_features(**inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    sim = (img_feat @ text_features.T).item()

    # Paint similarity onto heatmap
    heatmap[y1:y2, x1:x2] += sim
    count_map[y1:y2, x1:x2] += 1

# Normalize heatmap
count_map[count_map == 0] = 1  # avoid div by zero
heatmap = heatmap / count_map

# Convert to numpy and display
heatmap_np = heatmap.numpy()
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_pil)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(image_pil)
plt.imshow(heatmap_np, cmap='hot', alpha=0.5)
plt.colorbar()
plt.title("CLIP Similarity Heatmap")
plt.tight_layout()
plt.show()
