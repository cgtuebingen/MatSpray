#!/usr/bin/env python3
"""
Script to split a transforms.json file into train and test sets.
This is useful for NeRF-style datasets that need separate training and testing splits.
"""

import json
import random
import argparse
import os
import shutil
from pathlib import Path


def split_transforms(input_path, output_dir, train_ratio=0.8, seed=42, organize_images=True, images_dir=None):
    """
    Split transforms.json into train and test sets.
    
    Args:
        input_path: Path to the original transforms.json file
        output_dir: Directory to save the split files
        train_ratio: Ratio of frames to use for training (default: 0.8)
        seed: Random seed for reproducible splits (default: 42)
        organize_images: Whether to organize images into train/test folders (default: True)
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read the original transforms.json
    print(f"Reading transforms from: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    total_frames = len(frames)
    print(f"Total frames found: {total_frames}")
    
    # Shuffle frames to ensure random split
    shuffled_frames = frames.copy()
    random.shuffle(shuffled_frames)
    
    # Calculate split indices
    train_count = int(total_frames * train_ratio)
    test_count = total_frames - train_count
    
    # Split frames
    train_frames = shuffled_frames[:train_count]
    test_frames = shuffled_frames[train_count:]
    
    print(f"Train frames: {len(train_frames)} ({len(train_frames)/total_frames*100:.1f}%)")
    print(f"Test frames: {len(test_frames)} ({len(test_frames)/total_frames*100:.1f}%)")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create train and test directories for images
    train_images_dir = output_path / 'train' / 'images'
    test_images_dir = output_path / 'test' / 'images'
    
    if organize_images:
        train_images_dir.mkdir(parents=True, exist_ok=True)
        test_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created image directories:")
        print(f"  Train: {train_images_dir}")
        print(f"  Test: {test_images_dir}")
    
    # Update file paths and copy images
    if organize_images:
        print("\nOrganizing images...")
        
        # Get the source images directory
        if images_dir:
            source_images_dir = Path(images_dir)
        else:
            # Try to auto-detect: first check if images dir is in the same directory as transforms.json
            transforms_dir = Path(input_path).parent
            source_images_dir = transforms_dir / 'images'
            
            # If not found there, check current working directory
            if not source_images_dir.exists():
                source_images_dir = Path.cwd() / 'images'
            
            # If still not found, check parent of transforms directory
            if not source_images_dir.exists():
                source_images_dir = transforms_dir.parent / 'images'
        
        print(f"Looking for images in: {source_images_dir}")
        if not source_images_dir.exists():
            print(f"Warning: Source images directory {source_images_dir} not found!")
            print("Skipping image organization.")
            organize_images = False
        else:
            # Copy train images and update paths
            print("Copying train images...")
            for frame in train_frames:
                old_path = frame['file_path']
                # Extract filename from path like "./images/frame_00001.jpg"
                filename = Path(old_path).name
                new_path = f"./train/images/{filename}"
                
                # Copy the image
                src_file = source_images_dir / filename
                dst_file = train_images_dir / filename
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                else:
                    print(f"Warning: Source image {src_file} not found!")
                
                # Update the file path
                frame['file_path'] = new_path
            
            # Copy test images and update paths
            print("Copying test images...")
            for frame in test_frames:
                old_path = frame['file_path']
                # Extract filename from path like "./images/frame_00001.jpg"
                filename = Path(old_path).name
                new_path = f"./test/images/{filename}"
                
                # Copy the image
                src_file = source_images_dir / filename
                dst_file = test_images_dir / filename
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                else:
                    print(f"Warning: Source image {src_file} not found!")
                
                # Update the file path
                frame['file_path'] = new_path
            
            print(f"✓ Copied {len(train_frames)} train images and {len(test_frames)} test images")
    
    # Create train transforms
    train_data = {
        'camera_model': data['camera_model'],
        'orientation_override': data.get('orientation_override', 'none'),
        'frames': train_frames
    }
    
    train_file = output_path / 'transforms_train.json'
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=4)
    print(f"Train transforms saved to: {train_file}")
    
    # Create test transforms
    test_data = {
        'camera_model': data['camera_model'],
        'orientation_override': data.get('orientation_override', 'none'),
        'frames': test_frames
    }
    
    test_file = output_path / 'transforms_test.json'
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=4)
    print(f"Test transforms saved to: {test_file}")
    
    # Print some statistics about the split
    print("\nSplit Statistics:")
    print(f"Original file: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Train ratio: {train_ratio}")
    print(f"Random seed: {seed}")
    print(f"Images organized: {organize_images}")
    
    # Show some example frame names from each split
    print(f"\nExample train frames:")
    for i, frame in enumerate(train_frames[:5]):
        print(f"  {i+1}. {frame['file_path']}")
    if len(train_frames) > 5:
        print(f"  ... and {len(train_frames) - 5} more")
    
    print(f"\nExample test frames:")
    for i, frame in enumerate(test_frames[:5]):
        print(f"  {i+1}. {frame['file_path']}")
    if len(test_frames) > 5:
        print(f"  ... and {len(test_frames) - 5} more")
    
    return train_file, test_file


def main():
    parser = argparse.ArgumentParser(description='Split transforms.json into train and test sets')
    parser.add_argument('input_path', type=str, help='Path to the original transforms.json file')
    parser.add_argument('output_dir', type=str, help='Directory to save the split files')
    parser.add_argument('--train-ratio', type=float, default=0.8, 
                       help='Ratio of frames to use for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--no-images', action='store_true',
                       help='Skip organizing images into train/test folders')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='Path to images directory (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} does not exist!")
        return 1
    
    # Validate train ratio
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        print(f"Error: Train ratio must be between 0 and 1, got {args.train_ratio}")
        return 1
    
    try:
        train_file, test_file = split_transforms(
            args.input_path, 
            args.output_dir, 
            args.train_ratio, 
            args.seed,
            organize_images=not args.no_images,
            images_dir=args.images_dir
        )
        print(f"\n✓ Successfully split transforms.json!")
        print(f"✓ Train file: {train_file}")
        print(f"✓ Test file: {test_file}")
        return 0
        
    except Exception as e:
        print(f"Error during splitting: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 