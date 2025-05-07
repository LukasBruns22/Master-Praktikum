#!/usr/bin/env python3
# clip_patch_heatmap_multi_scale.py
# pip install torch transformers pillow matplotlib opencv-python

import argparse
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(
        description="Slide a square window (in patches) over an image, "
        "compute CLIP similarity to text, and build a per-patch heatmap."
    )
    p.add_argument("--image_path", type=str, default="ball.jpg")
    p.add_argument(
        "--num_patches_x",
        type=int,
        default=20,
        help="Number of patches across the image width"
    )
    p.add_argument(
        "--num_patches_y",
        type=int,
        default=20,
        help="Number of patches down the image height"
    )
    p.add_argument(
        "--stride_patches",
        type=int,
        default=1,
        help="Slide step in patches",
    )
    p.add_argument(
        "--prompt", type=str, required=True, help="Text prompt to compare against"
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = p.parse_args()
    print(f"Using device: {args.device} (CUDA available: {torch.cuda.is_available()})")

    img = Image.open(args.image_path).convert("RGB")
    img_w, img_h = img.size
    device = torch.device(args.device)

    # derive patch size based on number of patches
    args.patch_height = min(img_w // args.num_patches_x, img_h // args.num_patches_y)
    npx = img_w // args.patch_height
    npy = img_h // args.patch_height

    print(f"Image size: {img_w}x{img_h}")
    print(f"Patch size: {args.patch_height} px")
    print(f"Grid size: {npx}x{npy} patches")

    # CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device).eval()

    # text embedding
    txt = proc(text=[args.prompt], return_tensors="pt", padding=True)
    txt = {k: v.to(device) for k, v in txt.items() if k.startswith("input")}
    with torch.no_grad():
        t_emb = model.get_text_features(**txt)
        t_emb = t_emb / t_emb.norm(p=2, dim=-1, keepdim=True)

    # overall image → text similarity
    full = proc(images=img, return_tensors="pt", padding=True)
    full = {k: v.to(device) for k, v in full.items() if k.startswith("pixel")}
    with torch.no_grad():
        f_emb = model.get_image_features(**full)
        f_emb = f_emb / f_emb.norm(p=2, dim=-1, keepdim=True)
        overall_sim = float((f_emb @ t_emb.T).item())
    print(f"Overall CLIP similarity: {overall_sim:.4f}")

    # Multi-scale processing
    window_fractions = [0.10, 0.15, 0.20]
    combined_sum_map = np.zeros((npy, npx), dtype=np.float32)
    combined_cnt_map = np.zeros((npy, npx), dtype=np.int32)

    for window_fraction in window_fractions:
        window_size = max(1, int(min(npx, npy) * window_fraction))
        print(f"Processing window size: {window_size}x{window_size} patches ({int(window_fraction*100)}%)")

        for py in range(0, npy - window_size + 1, args.stride_patches):
            for px in range(0, npx - window_size + 1, args.stride_patches):
                x0, y0 = px * args.patch_height, py * args.patch_height
                win = img.crop((
                    x0, y0,
                    x0 + window_size * args.patch_height,
                    y0 + window_size * args.patch_height
                ))
                im = proc(images=win, return_tensors="pt", padding=True)
                im = {k: v.to(device) for k, v in im.items() if k.startswith("pixel")}
                with torch.no_grad():
                    i_emb = model.get_image_features(**im)
                    i_emb = i_emb / i_emb.norm(p=2, dim=-1, keepdim=True)
                    sim = float((i_emb @ t_emb.T).item())

                combined_sum_map[py:py + window_size, px:px + window_size] += sim
                combined_cnt_map[py:py + window_size, px:px + window_size] += 1

    # Final heatmap
    avg_map = combined_sum_map / np.maximum(combined_cnt_map, 1)
    heat = cv2.resize(avg_map, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    norm = np.uint8(255 * (heat - heat.min()) / (heat.max() - heat.min() + 1e-8))
    cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, cmap, 0.4, 0)

    # Save and show
    out = "overlay.png"
    cv2.imwrite(out, overlay)
    print(f"Saved overlay: {out}")

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Overlay (overall sim = {overall_sim:.4f})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
