#!/usr/bin/env python3
# clip_pure_scale_pyramid.py
# pip install torch transformers pillow matplotlib opencv-python scipy

import argparse
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Dict, Any


class CLIPPureScalePyramid:
    """
    Implements a pure scale pyramid approach for CLIP-based object localization.
    This approach completely eliminates sliding windows by directly computing
    similarities at different scales and intelligently combining them.
    """
    
    def __init__(self, device=None):
        """Initialize the CLIP model and processor"""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device} (CUDA available: {torch.cuda.is_available()})")
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device).eval()
    
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt using CLIP"""
        inputs = self.processor(text=[prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k.startswith("input")}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
        return text_features
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image using CLIP"""
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k.startswith("pixel")}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
        return image_features
    
    def compute_grid_similarities(self, 
                                image: Image.Image,
                                text_features: torch.Tensor,
                                grid_size: Tuple[int, int]) -> np.ndarray:
        """
        Divides image into a grid and computes similarity for each cell
        
        Args:
            image: PIL Image
            text_features: Encoded text features
            grid_size: (rows, cols) for the grid
            
        Returns:
            Similarity grid as numpy array
        """
        width, height = image.size
        rows, cols = grid_size
        
        cell_width = width // cols
        cell_height = height // rows
        
        similarities = np.zeros((rows, cols), dtype=np.float32)
        
        for r in range(rows):
            for c in range(cols):
                # Calculate cell boundaries
                x0 = c * cell_width
                y0 = r * cell_height
                x1 = min((c + 1) * cell_width, width)
                y1 = min((r + 1) * cell_height, height)
                
                # Extract cell
                cell = image.crop((x0, y0, x1, y1))
                
                # Resize to CLIP's expected input size
                cell = cell.resize((224, 224), Image.LANCZOS)
                
                # Get similarity
                cell_features = self.encode_image(cell)
                similarity = float((cell_features @ text_features.T).item())
                
                similarities[r, c] = similarity
                
        return similarities
    
    def create_pure_scale_pyramid(self,
                                 image: Image.Image,
                                 text_features: torch.Tensor,
                                 grid_sizes: List[Tuple[int, int]]) -> List[np.ndarray]:
        """
        Create pyramid of similarity maps at different grid resolutions
        
        Args:
            image: Original PIL Image
            text_features: Encoded text features
            grid_sizes: List of (rows, cols) tuples for different grid resolutions
            
        Returns:
            List of similarity maps at different resolutions
        """
        pyramid = []
        
        for grid_size in grid_sizes:
            rows, cols = grid_size
            print(f"Processing grid size: {rows}x{cols}")
            
            # Compute similarity grid
            sim_grid = self.compute_grid_similarities(image, text_features, grid_size)
            
            # Apply slight smoothing for better visualization
            sim_grid = gaussian_filter(sim_grid, sigma=0.7)
            
            pyramid.append((sim_grid, grid_size))
            
        return pyramid
    
    def upscale_and_combine_maps(self,
                               pyramid: List[Tuple[np.ndarray, Tuple[int, int]]],
                               target_size: Tuple[int, int]) -> np.ndarray:
        """
        Upscale and combine similarity maps from different grid sizes
        
        Args:
            pyramid: List of (similarity_map, grid_size) tuples
            target_size: Target size for the final heatmap (width, height)
            
        Returns:
            Combined similarity heatmap
        """
        width, height = target_size
        combined_map = np.zeros((height, width), dtype=np.float32)
        weights = []
        
        # Calculate weights for each scale - we'll give more weight to finer grids
        total_cells = sum(grid_size[0] * grid_size[1] for _, grid_size in pyramid)
        for _, grid_size in pyramid:
            grid_cells = grid_size[0] * grid_size[1]
            weight = grid_cells / total_cells
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        for i, ((sim_map, grid_size), weight) in enumerate(zip(pyramid, weights)):
            # Upscale similarity map to target size
            rows, cols = grid_size
            upscaled = cv2.resize(sim_map, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # Add to combined map with weight
            combined_map += weight * upscaled
        
        return combined_map
    
    def localize_object(self,
                      image_path: str,
                      prompt: str,
                      grid_sizes: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Localize an object in an image based on text prompt using a pure scale pyramid
        
        Args:
            image_path: Path to the input image
            prompt: Text description of the object to localize
            grid_sizes: List of grid sizes to process (default: [(2,2), (4,4), (8,8), (16,16)])
            
        Returns:
            Dictionary with results including the heatmap and original image
        """
        start_time = time.time()
        
        # Set default grid sizes if not provided
        if grid_sizes is None:
            grid_sizes = [(2, 2), (4, 4), (8, 8), (16, 16)]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        print(f"Image size: {width}x{height}")
        print(f"Text prompt: '{prompt}'")
        
        # Encode text
        text_features = self.encode_text(prompt)
        
        # Get full image similarity (baseline)
        full_image_features = self.encode_image(image)
        full_similarity = float((full_image_features @ text_features.T).item())
        print(f"Overall CLIP similarity: {full_similarity:.4f}")
        
        # Create pure scale pyramid
        pyramid = self.create_pure_scale_pyramid(image, text_features, grid_sizes)
        
        # Combine maps from pyramid
        heatmap = self.upscale_and_combine_maps(pyramid, (width, height))
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        return {
            "image": np.array(image),
            "heatmap": heatmap,
            "overall_similarity": full_similarity,
            "processing_time": processing_time
        }
    
    def visualize_results(self,
                        results: Dict[str, Any],
                        save_path: str = None,
                        show_plot: bool = True) -> None:
        """
        Visualize the localization results
        
        Args:
            results: Results dictionary from localize_object
            save_path: Path to save the visualization (default: None)
            show_plot: Whether to display the plot (default: True)
        """
        image = results["image"]
        heatmap = results["heatmap"]
        similarity = results["overall_similarity"]
        processing_time = results["processing_time"]
        
        # Create color heatmap
        norm_heatmap = np.uint8(255 * heatmap)
        color_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
        
        # Create overlay
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, color_heatmap, 0.4, 0)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")
        
        # Heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB))
        plt.title("Similarity Heatmap")
        plt.axis("off")
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"Overlay (similarity: {similarity:.4f})")
        plt.axis("off")
        
        plt.suptitle(f"Processing time: {processing_time:.2f} seconds", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            cv2.imwrite(f"{save_path}_overlay.png", overlay)
            print(f"Saved visualization to {save_path}")
        
        if show_plot:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Localize objects in images using CLIP with pure scale pyramid approach"
    )
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to localize")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save visualization")
    parser.add_argument("--grid_level", type=int, default=4, 
                        help="Maximum grid level (e.g., 4 means up to 2^4x2^4 grid)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Calculate grid sizes based on grid level
    grid_sizes = [(2**i, 2**i) for i in range(1, args.grid_level + 1)]
    
    # Initialize localizer
    localizer = CLIPPureScalePyramid(device=args.device)
    
    # Localize object
    results = localizer.localize_object(
        image_path=args.image_path,
        prompt=args.prompt,
        grid_sizes=grid_sizes
    )
    
    # Visualize results
    localizer.visualize_results(results, save_path=args.output_path)


if __name__ == "__main__":
    main()