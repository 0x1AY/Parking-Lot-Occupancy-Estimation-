#!/usr/bin/env python3
"""
Convert APKLOT dataset to YOLO segmentation format.

This script converts the APKLOT parking lot dataset from LabelMe JSON format
to YOLO segmentation format for training YOLOv11 models.

Usage:
    python tools/convert_apklot_to_yolo.py

Input: APKLOT/1. Satellite/Dataset/
Output: datasets/apklot/
"""

import json
import base64
import shutil
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm


def setup_directories(output_base: Path):
    """Create YOLO dataset directory structure."""
    dirs = [
        output_base / 'images' / 'train',
        output_base / 'images' / 'val',
        output_base / 'labels' / 'train',
        output_base / 'labels' / 'val',
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
    
    return dirs


def load_split_files(split_dir: Path):
    """Load train and validation split file lists."""
    train_path = split_dir / 'train.txt'
    val_path = split_dir / 'val.txt'
    
    train_files = []
    val_files = []
    
    if train_path.exists():
        with open(train_path, 'r') as f:
            train_files = [line.strip() for line in f if line.strip()]
    
    if val_path.exists():
        with open(val_path, 'r') as f:
            val_files = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(train_files)} training files")
    print(f"Loaded {len(val_files)} validation files")
    
    return train_files, val_files


def convert_labelme_to_yolo(json_path: Path, image_output: Path, label_output: Path):
    """
    Convert a single LabelMe JSON file to YOLO segmentation format.
    
    Args:
        json_path: Path to LabelMe JSON file
        image_output: Path to save extracted image
        label_output: Path to save YOLO label file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract and save image from base64
        if 'imageData' in data and data['imageData']:
            try:
                img_data = base64.b64decode(data['imageData'])
                img = Image.open(io.BytesIO(img_data))
                
                # Save as JPG
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(image_output, 'JPEG', quality=95)
                
            except Exception as e:
                print(f"Warning: Could not extract image from {json_path.name}: {e}")
                return False
        else:
            print(f"Warning: No imageData in {json_path.name}")
            return False
        
        # Get image dimensions
        img_w, img_h = img.size
        
        # Convert polygons to YOLO format
        yolo_labels = []
        
        for shape in data.get('shapes', []):
            # Filter for parking lot class (label "1" in APKLOT)
            if str(shape.get('label')) == '1':
                points = shape.get('points', [])
                
                if len(points) < 3:
                    continue  # Skip invalid polygons
                
                # Normalize coordinates to 0-1 range
                normalized_points = []
                for x, y in points:
                    x_norm = max(0.0, min(1.0, x / img_w))
                    y_norm = max(0.0, min(1.0, y / img_h))
                    normalized_points.append(f"{x_norm:.6f}")
                    normalized_points.append(f"{y_norm:.6f}")
                
                # YOLO format: class_id x1 y1 x2 y2 ... xn yn
                yolo_line = f"0 {' '.join(normalized_points)}"
                yolo_labels.append(yolo_line)
        
        # Save YOLO label file
        with open(label_output, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        return True
        
    except Exception as e:
        print(f"Error converting {json_path.name}: {e}")
        return False


def convert_pascal_to_yolo(image_source: Path, image_output: Path, 
                          label_output: Path, xml_path: Path = None):
    """
    Convert Pascal VOC format to YOLO by copying image and creating empty/simple label.
    This is a fallback if LabelMe conversion fails.
    
    Args:
        image_source: Source image path (JPG)
        image_output: Destination image path
        label_output: Destination label path
        xml_path: Optional XML annotation path (not used for segmentation)
    """
    try:
        # Copy image
        shutil.copy2(image_source, image_output)
        
        # Create empty label file (will be populated if we parse XML)
        label_output.touch()
        
        return True
        
    except Exception as e:
        print(f"Error copying {image_source.name}: {e}")
        return False


def main():
    """Main conversion function."""
    print("=" * 70)
    print("APKLOT to YOLO Segmentation Converter")
    print("=" * 70)
    print()
    
    # Define paths
    apklot_base = Path("/Users/ay/Desktop/deeplearning/Parking lot /APKLOT")
    labelme_dir = apklot_base / "1. Satellite" / "Dataset" / "labelme_20"
    pascal_base = apklot_base / "1. Satellite" / "Dataset" / "World" / "PASCAL_format"
    split_dir = pascal_base / "ImageSets" / "Segmentation"
    
    output_base = Path("/Users/ay/Desktop/deeplearning/Parking lot /Parking-Lot-Occupancy-Estimation-/datasets/apklot")
    
    # Verify input directories exist
    if not apklot_base.exists():
        print(f"❌ Error: APKLOT directory not found: {apklot_base}")
        return
    
    if not labelme_dir.exists():
        print(f"❌ Error: LabelMe directory not found: {labelme_dir}")
        return
    
    print(f"✓ APKLOT base: {apklot_base}")
    print(f"✓ LabelMe dir: {labelme_dir}")
    print(f"✓ Output dir: {output_base}")
    print()
    
    # Create output directory structure
    print("Creating output directories...")
    setup_directories(output_base)
    print()
    
    # Load train/val splits
    print("Loading train/val splits...")
    train_files, val_files = load_split_files(split_dir)
    print()
    
    # Convert training set
    print("Converting training set...")
    train_success = 0
    train_failed = 0
    
    for filename in tqdm(train_files, desc="Train"):
        json_path = labelme_dir / f"{filename}.json"
        image_output = output_base / 'images' / 'train' / f"{filename}.jpg"
        label_output = output_base / 'labels' / 'train' / f"{filename}.txt"
        
        if json_path.exists():
            if convert_labelme_to_yolo(json_path, image_output, label_output):
                train_success += 1
            else:
                train_failed += 1
        else:
            print(f"Warning: JSON not found: {json_path.name}")
            train_failed += 1
    
    print(f"✓ Training: {train_success} success, {train_failed} failed")
    print()
    
    # Convert validation set
    print("Converting validation set...")
    val_success = 0
    val_failed = 0
    
    for filename in tqdm(val_files, desc="Val"):
        json_path = labelme_dir / f"{filename}.json"
        image_output = output_base / 'images' / 'val' / f"{filename}.jpg"
        label_output = output_base / 'labels' / 'val' / f"{filename}.txt"
        
        if json_path.exists():
            if convert_labelme_to_yolo(json_path, image_output, label_output):
                val_success += 1
            else:
                val_failed += 1
        else:
            print(f"Warning: JSON not found: {json_path.name}")
            val_failed += 1
    
    print(f"✓ Validation: {val_success} success, {val_failed} failed")
    print()
    
    # Summary
    print("=" * 70)
    print("Conversion Summary")
    print("=" * 70)
    print(f"Training set:   {train_success}/{len(train_files)} converted")
    print(f"Validation set: {val_success}/{len(val_files)} converted")
    print(f"Total:          {train_success + val_success}/{len(train_files) + len(val_files)} converted")
    print()
    print(f"Output location: {output_base}")
    print()
    
    # Verify some samples
    print("Verifying conversion...")
    sample_image = output_base / 'images' / 'train' / f"{train_files[0]}.jpg"
    sample_label = output_base / 'labels' / 'train' / f"{train_files[0]}.txt"
    
    if sample_image.exists() and sample_label.exists():
        img = Image.open(sample_image)
        with open(sample_label, 'r') as f:
            labels = f.readlines()
        
        print(f"✓ Sample image: {sample_image.name}")
        print(f"  - Size: {img.size}")
        print(f"  - Mode: {img.mode}")
        print(f"  - Labels: {len(labels)} parking lot(s)")
        
        if labels:
            first_label = labels[0].strip().split()
            print(f"  - First polygon: class={first_label[0]}, points={len(first_label[1:])//2}")
    else:
        print("⚠ Could not verify sample files")
    
    print()
    print("✅ Conversion complete!")
    print()
    print("Next steps:")
    print("1. Create data/apklot.yaml configuration file")
    print("2. Train with: yolo segment train data=data/apklot.yaml model=yolov11m-seg.pt")


if __name__ == '__main__':
    main()
