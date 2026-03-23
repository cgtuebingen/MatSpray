#!/usr/bin/env python3
"""
Script to separate a metallic-roughness image into separate metallic and roughness images.
- R channel (index 0) = metallic
- G channel (index 1) = roughness
"""

import sys
import os
import numpy as np
from PIL import Image


def split_metallic_roughness(input_path, metallic_output=None, roughness_output=None):
    """
    Split a metallic-roughness image into separate metallic and roughness images.
    
    Args:
        input_path: Path to the input metallic-roughness image
        metallic_output: Path for the output metallic image (optional)
        roughness_output: Path for the output roughness image (optional)
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        abs_path = os.path.abspath(input_path)
        print(f"Error: Input file does not exist: {input_path}")
        print(f"Absolute path: {abs_path}")
        sys.exit(1)
    
    # Load the image
    try:
        img = Image.open(input_path)
        print(f"Original image mode: {img.mode}, size: {img.size}")
        
        # Convert to numpy array based on mode to avoid any alpha compositing
        if img.mode == 'RGBA':
            # For RGBA, extract channels directly from numpy array
            rgba_array = np.array(img, dtype=np.uint8)
            print(f"Loaded RGBA image, shape: {rgba_array.shape}")
            
            # Check if alpha channel exists and has variation
            alpha = rgba_array[:, :, 3].astype(np.float32) / 255
            alpha_min, alpha_max = alpha.min(), alpha.max()
            print(f"Alpha channel range: [{alpha_min:.3f}, {alpha_max:.3f}]")
            
            # Extract RGB channels (indices 0, 1, 2)
            rgb_array = rgba_array[:, :, :3].astype(np.float32)
            
            # Check if image appears to be premultiplied alpha
            # If alpha < 1 and RGB values are low, it might be premultiplied
            # Try un-premultiplying if alpha is not 1.0
            if alpha_max < 1.0:
                print("Warning: Alpha channel has values < 1.0, checking for premultiplied alpha...")
                # Un-premultiply: RGB = RGB_premult / alpha (avoid division by zero)
                alpha_safe = np.maximum(alpha, 1.0/255.0)  # Avoid division by zero
                rgb_array = rgb_array * alpha_safe[:, :, np.newaxis]
                rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
                print("Attempted to un-premultiply alpha")
            else:
                rgb_array = rgb_array.astype(np.uint8) * alpha[:, :, np.newaxis]
            
            img_array = rgb_array.copy()  # Use copy to ensure contiguous
        elif img.mode == 'RGB':
            img_array = np.array(img, dtype=np.uint8)
            print(f"Loaded RGB image, shape: {img_array.shape}")
        elif img.mode == 'LA' or img.mode == 'P':
            # Convert to RGB first for these modes
            img = img.convert('RGB')
            img_array = np.array(img, dtype=np.uint8)
            print(f"Converted {img.mode} to RGB, shape: {img_array.shape}")
        else:
            # For other modes, convert to RGB
            img = img.convert('RGB')
            img_array = np.array(img, dtype=np.uint8)
            print(f"Converted to RGB, shape: {img_array.shape}")
        
        # Debug: check edge pixel values
        print(f"Edge pixel (0,0) RGB: {img_array[0, 0, :]}")
        print(f"Edge pixel (0,-1) RGB: {img_array[0, -1, :]}")
        print(f"Edge pixel (-1,0) RGB: {img_array[-1, 0, :]}")
        print(f"Edge pixel (-1,-1) RGB: {img_array[-1, -1, :]}")
        print(f"Center pixel RGB: {img_array[img_array.shape[0]//2, img_array.shape[1]//2, :]}")
        
    except Exception as e:
        print(f"Error loading image '{input_path}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Extract channels - use direct indexing to avoid any issues
    # R channel (index 0) = metallic
    metallic_channel = img_array[:, :, 0].copy()  # Extract as 2D array
    metallic_3channel = np.stack([metallic_channel, metallic_channel, metallic_channel], axis=2).astype(np.uint8)
    
    # G channel (index 1) = roughness
    roughness_channel = img_array[:, :, 1].copy()  # Extract as 2D array
    roughness_3channel = np.stack([roughness_channel, roughness_channel, roughness_channel], axis=2).astype(np.uint8)
    
    # Debug: check extracted channel values at edges
    print(f"\nMetallic channel - Edge (0,0): {metallic_channel[0, 0]}, Edge (0,-1): {metallic_channel[0, -1]}")
    print(f"Roughness channel - Edge (0,0): {roughness_channel[0, 0]}, Edge (0,-1): {roughness_channel[0, -1]}")
    
    # Generate output paths if not provided
    base_name = os.path.splitext(input_path)[0]
    if metallic_output is None:
        metallic_output = base_name + '_metallic.png'
    if roughness_output is None:
        roughness_output = base_name + '_roughness.png'
    
    # Ensure .png extension
    if not metallic_output.lower().endswith('.png'):
        metallic_output = metallic_output + '.png'
    if not roughness_output.lower().endswith('.png'):
        roughness_output = roughness_output + '.png'
    
    # Save metallic image
    metallic_img = Image.fromarray(metallic_3channel, mode='RGB')
    # Verify before saving
    saved_array = np.array(metallic_img, dtype=np.uint8)
    print(f"\nBefore saving metallic - Edge (0,0): {saved_array[0, 0, :]}, Edge (0,-1): {saved_array[0, -1, :]}")
    metallic_img.save(metallic_output, 'PNG')
    print(f"Saved metallic image: {metallic_output}")
    
    # Verify saved file
    verify_img = Image.open(metallic_output)
    verify_array = np.array(verify_img, dtype=np.uint8)
    print(f"After saving metallic - Edge (0,0): {verify_array[0, 0, :]}, Edge (0,-1): {verify_array[0, -1, :]}")
    
    # Save roughness image
    roughness_img = Image.fromarray(roughness_3channel, mode='RGB')
    # Verify before saving
    saved_array = np.array(roughness_img, dtype=np.uint8)
    print(f"\nBefore saving roughness - Edge (0,0): {saved_array[0, 0, :]}, Edge (0,-1): {saved_array[0, -1, :]}")
    roughness_img.save(roughness_output, 'PNG')
    print(f"Saved roughness image: {roughness_output}")
    
    # Verify saved file
    verify_img = Image.open(roughness_output)
    verify_array = np.array(verify_img, dtype=np.uint8)
    print(f"After saving roughness - Edge (0,0): {verify_array[0, 0, :]}, Edge (0,-1): {verify_array[0, -1, :]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluation/split_metallic_roughness.py <input_metallicroughness.png> [metallic_output.png] [roughness_output.png]")
        print("\nExamples:")
        print("  python evaluation/split_metallic_roughness.py input_metallicroughness.png")
        print("  python evaluation/split_metallic_roughness.py input_metallicroughness.png metallic.png roughness.png")
        sys.exit(1)
    
    input_path = sys.argv[1]
    metallic_output = sys.argv[2] if len(sys.argv) > 2 else None
    roughness_output = sys.argv[3] if len(sys.argv) > 3 else None
    
    split_metallic_roughness(input_path, metallic_output, roughness_output)

