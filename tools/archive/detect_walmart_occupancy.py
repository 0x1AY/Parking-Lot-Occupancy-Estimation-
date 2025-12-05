#!/usr/bin/env python3
"""
Run occupancy detection on Walmart parking lot images.
Uses the trained multi-class YOLOv11m model to detect cars and stalls,
then calculates occupancy rates for each location.
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Configuration
MODEL_PATH = "parking_runs/yolo11m_multiclass/weights/best.pt"
WALMART_IMAGES_DIR = "walmart_locations/images_hires"  # High-res images from Google Maps
OUTPUT_DIR = "walmart_locations/results"
WALMART_CSV = "walmart lots.csv"

# Detection parameters
CONF_THRESHOLD = 0.25  # Lower threshold for better recall
IOU_THRESHOLD = 0.3    # Overlap threshold for matching cars to stalls

CLASS_NAMES = {
    0: 'car',
    1: 'lot_boundary',
    3: 'stall'
}


def calculate_overlap_ratio(car_box, stall_box):
    """
    Calculate what fraction of the car overlaps with the stall.
    
    Args:
        car_box: Car bounding box [x1, y1, x2, y2]
        stall_box: Stall bounding box [x1, y1, x2, y2]
    
    Returns:
        Overlap ratio (0 to 1)
    """
    x1_min, y1_min, x1_max, y1_max = car_box
    x2_min, y2_min, x2_max, y2_max = stall_box
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    car_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    overlap_ratio = inter_area / car_area if car_area > 0 else 0.0
    
    return overlap_ratio


def detect_occupancy(model, image_path, conf_threshold, iou_threshold):
    """
    Detect parking lot occupancy in an image.
    
    Returns:
        Dictionary with occupancy data
    """
    # Run detection
    results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)
    result = results[0]
    
    # Separate detections by class
    cars = []
    stalls = []
    
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        
        if cls_id == 0:  # car
            cars.append({
                'bbox': xyxy,
                'conf': conf
            })
        elif cls_id == 3:  # stall
            stalls.append({
                'bbox': xyxy,
                'conf': conf,
                'occupied': False,
                'car_idx': None
            })
    
    # Match cars to stalls
    occupied_stalls = []
    vacant_stalls = []
    
    for stall_idx, stall in enumerate(stalls):
        max_overlap = 0.0
        matched_car_idx = None
        
        for car_idx, car in enumerate(cars):
            overlap = calculate_overlap_ratio(car['bbox'], stall['bbox'])
            
            if overlap > max_overlap and overlap >= iou_threshold:
                max_overlap = overlap
                matched_car_idx = car_idx
        
        if matched_car_idx is not None:
            stall['occupied'] = True
            stall['car_idx'] = matched_car_idx
            occupied_stalls.append(stall_idx)
        else:
            vacant_stalls.append(stall_idx)
    
    # Calculate statistics
    total_stalls = len(stalls)
    occupied_count = len(occupied_stalls)
    occupancy_rate = (occupied_count / total_stalls * 100) if total_stalls > 0 else 0.0
    
    return {
        'cars': cars,
        'stalls': stalls,
        'occupied_stalls': occupied_stalls,
        'vacant_stalls': vacant_stalls,
        'occupancy_rate': occupancy_rate,
        'total_stalls': total_stalls,
        'occupied_count': occupied_count,
        'vacant_count': len(vacant_stalls),
        'total_cars_detected': len(cars)
    }


def visualize_and_save(image_path, occupancy_data, output_path):
    """
    Visualize occupancy detection and save to file.
    """
    # Read image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(img)
    
    # Draw stalls
    for stall in occupancy_data['stalls']:
        x1, y1, x2, y2 = stall['bbox']
        width = x2 - x1
        height = y2 - y1
        
        # Color based on occupancy
        if stall['occupied']:
            color = '#FF0000'  # Red
            label = 'Occupied'
        else:
            color = '#00FF00'  # Green
            label = 'Vacant'
        
        rect = plt.Rectangle((x1, y1), width, height,
                            fill=False, edgecolor=color, linewidth=3)
        ax.add_patch(rect)
        
        ax.text(x1 + 5, y1 + 20, label,
               color='white', fontsize=8, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    # Draw cars (dashed yellow)
    for car in occupancy_data['cars']:
        x1, y1, x2, y2 = car['bbox']
        width = x2 - x1
        height = y2 - y1
        
        rect = plt.Rectangle((x1, y1), width, height,
                            fill=False, edgecolor='#FFFF00',
                            linewidth=2, linestyle='--')
        ax.add_patch(rect)
    
    # Add statistics
    stats_text = f"Total Stalls: {occupancy_data['total_stalls']}\\n"
    stats_text += f"Occupied: {occupancy_data['occupied_count']}\\n"
    stats_text += f"Vacant: {occupancy_data['vacant_count']}\\n"
    stats_text += f"Cars Detected: {occupancy_data['total_cars_detected']}\\n"
    stats_text += f"Occupancy: {occupancy_data['occupancy_rate']:.1f}%"
    
    ax.text(10, 50, stats_text,
           color='white', fontsize=12, weight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='black', alpha=0.7),
           verticalalignment='top')
    
    ax.axis('off')
    plt.title(f"Walmart Occupancy: {os.path.basename(image_path)}", 
             fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("="*70)
    print("Walmart Parking Lot Occupancy Detection")
    print("="*70)
    print()
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Please train the multi-class model first.")
        return
    
    print(f"📦 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully")
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all Walmart images
    images_dir = Path(WALMART_IMAGES_DIR)
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        print("   Please run tools/download_walmart_images.py first.")
        return
    
    image_files = sorted(images_dir.glob("walmart_*.png"))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    print(f"📸 Found {len(image_files)} Walmart images")
    print()
    
    # Process each image
    all_results = []
    
    for idx, img_path in enumerate(image_files, 1):
        location_name = img_path.stem.replace('walmart_', '').replace('_', ' ')
        
        print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
        
        # Detect occupancy
        occupancy_data = detect_occupancy(model, img_path, CONF_THRESHOLD, IOU_THRESHOLD)
        occupancy_data['location'] = location_name
        occupancy_data['image_file'] = img_path.name
        
        # Save visualization
        output_img = output_dir / f"{img_path.stem}_occupancy.jpg"
        visualize_and_save(img_path, occupancy_data, output_img)
        
        # Print results
        print(f"   Stalls: {occupancy_data['total_stalls']} | "
              f"Occupied: {occupancy_data['occupied_count']} | "
              f"Vacant: {occupancy_data['vacant_count']} | "
              f"Rate: {occupancy_data['occupancy_rate']:.1f}%")
        print(f"   💾 Saved: {output_img.name}")
        print()
        
        # Store for summary
        all_results.append({
            'location': location_name,
            'image_file': img_path.name,
            'total_stalls': occupancy_data['total_stalls'],
            'occupied': occupancy_data['occupied_count'],
            'vacant': occupancy_data['vacant_count'],
            'occupancy_rate': occupancy_data['occupancy_rate'],
            'cars_detected': occupancy_data['total_cars_detected']
        })
    
    # Save results to JSON
    results_json = output_dir / "walmart_occupancy_results.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("="*70)
    print("📊 SUMMARY - WALMART PARKING LOT OCCUPANCY")
    print("="*70)
    
    total_stalls = sum(r['total_stalls'] for r in all_results)
    total_occupied = sum(r['occupied'] for r in all_results)
    total_vacant = sum(r['vacant'] for r in all_results)
    avg_occupancy = (total_occupied / total_stalls * 100) if total_stalls > 0 else 0.0
    
    print(f"Locations analyzed:  {len(all_results)}")
    print(f"Total stalls:        {total_stalls}")
    print(f"Total occupied:      {total_occupied} ({avg_occupancy:.1f}%)")
    print(f"Total vacant:        {total_vacant} ({100-avg_occupancy:.1f}%)")
    print()
    
    print("Per-location breakdown:")
    print("-" * 70)
    for r in all_results:
        print(f"{r['location'][:45]:45s} | Stalls: {r['total_stalls']:3d} | "
              f"Occupied: {r['occupied']:3d} ({r['occupancy_rate']:5.1f}%)")
    print("="*70)
    
    print(f"\\n💾 Results saved to: {results_json}")
    print(f"📁 Visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
