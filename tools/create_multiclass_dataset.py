#!/usr/bin/env python3
"""
Create a multi-class dataset from Dataset-V1-detect containing only images
with both car (class 0) and stall (class 3) annotations.
Optionally includes lot_boundary (class 1) if present.

Output: Dataset-V1-multiclass with filtered train/valid/test splits
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

# Paths
SOURCE_DATASET = "Dataset-V1-detect"
TARGET_DATASET = "Dataset-V1-multiclass"

# Classes we want
REQUIRED_CLASSES = {0, 3}  # car, stall
OPTIONAL_CLASSES = {1}     # lot_boundary
ALL_CLASSES = REQUIRED_CLASSES | OPTIONAL_CLASSES


def parse_label_file(label_path):
    """Parse YOLO label file and return set of classes present."""
    classes = set()
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    return classes


def has_required_classes(classes):
    """Check if label has both car and stall classes."""
    return REQUIRED_CLASSES.issubset(classes)


def filter_dataset_split(split_name):
    """Filter one split (train/valid/test) and copy matching images/labels."""
    source_images = Path(SOURCE_DATASET) / split_name / "images"
    source_labels = Path(SOURCE_DATASET) / split_name / "labels"
    
    target_images = Path(TARGET_DATASET) / split_name / "images"
    target_labels = Path(TARGET_DATASET) / split_name / "labels"
    
    # Create target directories
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {
        'total': 0,
        'filtered': 0,
        'has_car': 0,
        'has_stall': 0,
        'has_lot_boundary': 0,
    }
    
    # Process all label files
    if not source_labels.exists():
        print(f"⚠️  Warning: {source_labels} does not exist, skipping {split_name}")
        return stats
    
    for label_file in source_labels.glob("*.txt"):
        stats['total'] += 1
        
        # Parse classes in this label file
        classes = parse_label_file(label_file)
        
        # Count class occurrences
        if 0 in classes:
            stats['has_car'] += 1
        if 3 in classes:
            stats['has_stall'] += 1
        if 1 in classes:
            stats['has_lot_boundary'] += 1
        
        # Check if it has required classes
        if has_required_classes(classes):
            stats['filtered'] += 1
            
            # Copy label file
            target_label = target_labels / label_file.name
            shutil.copy2(label_file, target_label)
            
            # Copy corresponding image
            image_name = label_file.stem + ".jpg"
            source_image = source_images / image_name
            target_image = target_images / image_name
            
            if source_image.exists():
                # Resolve symlink if needed
                if source_image.is_symlink():
                    source_image = source_image.resolve()
                shutil.copy2(source_image, target_image)
            else:
                print(f"⚠️  Warning: Image not found: {source_image}")
    
    return stats


def create_data_yaml():
    """Create data.yaml for the new dataset."""
    yaml_content = f"""# Multi-class parking lot dataset
# Classes: car, lot_boundary, stall
# Filtered from Dataset-V1-detect to include only images with both car and stall annotations

path: .  # dataset root dir (relative path)
train: train/images
val: valid/images
test: test/images

# Classes
nc: 4  # number of classes
names:
  0: car
  1: lot_boundary
  2: objects
  3: stall
"""
    
    yaml_path = Path(TARGET_DATASET) / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"✅ Created {yaml_path}")


def main():
    print("="*60)
    print("Creating Multi-Class Dataset")
    print("="*60)
    print(f"Source: {SOURCE_DATASET}")
    print(f"Target: {TARGET_DATASET}")
    print(f"Required classes: car (0), stall (3)")
    print(f"Optional classes: lot_boundary (1)")
    print("="*60)
    print()
    
    # Create target dataset directory
    Path(TARGET_DATASET).mkdir(exist_ok=True)
    
    # Process each split
    total_stats = defaultdict(int)
    
    for split in ['train', 'valid', 'test']:
        print(f"📁 Processing {split} split...")
        stats = filter_dataset_split(split)
        
        # Accumulate stats
        for key, value in stats.items():
            total_stats[key] += value
        
        # Print split stats
        print(f"   Total images: {stats['total']}")
        print(f"   Has car (0): {stats['has_car']}")
        print(f"   Has stall (3): {stats['has_stall']}")
        print(f"   Has lot_boundary (1): {stats['has_lot_boundary']}")
        print(f"   ✅ Filtered (car+stall): {stats['filtered']}")
        print()
    
    # Create data.yaml
    create_data_yaml()
    
    # Print summary
    print("="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Total images processed: {total_stats['total']}")
    print(f"Images with car: {total_stats['has_car']} ({100*total_stats['has_car']/total_stats['total']:.1f}%)")
    print(f"Images with stall: {total_stats['has_stall']} ({100*total_stats['has_stall']/total_stats['total']:.1f}%)")
    print(f"Images with lot_boundary: {total_stats['has_lot_boundary']} ({100*total_stats['has_lot_boundary']/total_stats['total']:.1f}%)")
    print(f"✅ Images in new dataset: {total_stats['filtered']} ({100*total_stats['filtered']/total_stats['total']:.1f}%)")
    print("="*60)
    print(f"\n✨ Dataset created: {TARGET_DATASET}")
    print(f"   Ready for multi-class training (car, stall, lot_boundary)")


if __name__ == "__main__":
    main()
