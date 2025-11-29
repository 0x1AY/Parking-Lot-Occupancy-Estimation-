#!/usr/bin/env python3
"""
Visualize APKLOT YOLO segmentation dataset samples.

Usage:
    python tools/visualize_apklot.py [--split train|val] [--num_samples 5]
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import random


def load_yolo_segmentation(label_path: Path):
    """Load YOLO segmentation polygons from label file."""
    polygons = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + at least 3 points (6 coords)
                continue
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Group into (x, y) pairs
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    points.append((coords[i], coords[i + 1]))
            
            polygons.append({
                'class_id': class_id,
                'points': points
            })
    
    return polygons


def draw_polygons(image: np.ndarray, polygons: list, color=(0, 255, 0), thickness=2):
    """Draw segmentation polygons on image."""
    img_h, img_w = image.shape[:2]
    overlay = image.copy()
    
    for poly in polygons:
        # Convert normalized coords to pixel coords
        points = []
        for x_norm, y_norm in poly['points']:
            x = int(x_norm * img_w)
            y = int(y_norm * img_h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Draw filled polygon with transparency
        cv2.fillPoly(overlay, [points], color)
        
        # Draw polygon outline
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 255), thickness=thickness)
    
    # Blend overlay with original
    alpha = 0.3
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return image


def visualize_samples(dataset_path: Path, split: str = 'train', num_samples: int = 5):
    """Visualize random samples from dataset."""
    images_dir = dataset_path / 'images' / split
    labels_dir = dataset_path / 'labels' / split
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg'))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    # Sample random images
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"Visualizing {len(samples)} samples from {split} set...")
    print(f"Total images available: {len(image_files)}")
    print()
    
    for i, img_path in enumerate(samples, 1):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠ Could not load {img_path.name}")
            continue
        
        # Load corresponding label
        label_path = labels_dir / img_path.with_suffix('.txt').name
        
        if not label_path.exists():
            print(f"⚠ No label found for {img_path.name}")
            continue
        
        polygons = load_yolo_segmentation(label_path)
        
        # Draw polygons
        vis_image = draw_polygons(image, polygons)
        
        # Add info text
        info_text = f"{img_path.name} | {len(polygons)} parking lot(s)"
        cv2.putText(vis_image, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Calculate statistics
        img_h, img_w = image.shape[:2]
        total_area = 0
        for poly in polygons:
            points = np.array([[int(x * img_w), int(y * img_h)] 
                             for x, y in poly['points']], dtype=np.int32)
            area = cv2.contourArea(points)
            total_area += area
        
        coverage = (total_area / (img_h * img_w)) * 100
        coverage_text = f"Coverage: {coverage:.1f}%"
        cv2.putText(vis_image, coverage_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save visualization
        output_dir = dataset_path / 'visualizations' / split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"vis_{img_path.name}"
        cv2.imwrite(str(output_path), vis_image)
        
        print(f"✓ [{i}/{len(samples)}] {img_path.name}")
        print(f"  - Size: {img_w}x{img_h}")
        print(f"  - Parking lots: {len(polygons)}")
        print(f"  - Coverage: {coverage:.1f}%")
        print(f"  - Saved: {output_path}")
        print()
    
    print(f"✅ Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize APKLOT YOLO dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                       help='Dataset split to visualize')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    dataset_path = Path("/Users/ay/Desktop/deeplearning/Parking lot /Parking-Lot-Occupancy-Estimation-/datasets/apklot")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print("=" * 70)
    print("APKLOT Dataset Visualization")
    print("=" * 70)
    print()
    
    visualize_samples(dataset_path, args.split, args.num_samples)


if __name__ == '__main__':
    main()
