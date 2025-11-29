#!/usr/bin/env python3
"""
Train YOLOv11 segmentation model on APKLOT dataset for parking lot localization.

This is Stage 1 of the multi-stage parking lot occupancy detection pipeline.
The model learns to detect and segment parking lot boundaries in wide-area
satellite imagery.

Usage:
    python tools/train_apklot_stage1.py
"""

from ultralytics import YOLO
from pathlib import Path


def main():
    """Train YOLOv11-seg on APKLOT dataset."""
    print("=" * 70)
    print("Training YOLOv11-seg on APKLOT Dataset")
    print("Stage 1: Parking Lot Localization")
    print("=" * 70)
    print()
    
    # Check if data config exists
    data_yaml = Path("data/apklot.yaml")
    if not data_yaml.exists():
        print(f"❌ Error: Dataset config not found: {data_yaml}")
        return
    
    print(f"✓ Dataset config: {data_yaml}")
    print()
    
    # Initialize model
    print("Loading YOLOv11m-seg pretrained model...")
    print("(Model will be downloaded automatically if not found)")
    try:
        model = YOLO('yolo11m-seg.pt')  # Updated model name
        print("✓ Model loaded")
    except Exception as e:
        print(f"Note: {e}")
        print("Downloading YOLOv11m-seg weights...")
        model = YOLO('yolo11m-seg.pt')
        print("✓ Model loaded")
    print()
    
    # Training configuration
    print("Training configuration:")
    config = {
        'data': 'data/apklot.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'project': 'parking_lot_localization',
        'name': 'apklot_stage1',
        'patience': 20,
        'save_period': 10,
        'device': 'mps',  # Use Metal Performance Shaders on macOS
        'workers': 8,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'single_cls': True,  # Single class (parking_lot)
        'rect': False,  # Rectangular training
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 10,  # Close mosaic augmentation in last N epochs
        'amp': True,  # Automatic mixed precision
        'fraction': 1.0,  # Use 100% of dataset
        'overlap_mask': True,  # Masks can overlap
        'mask_ratio': 4,  # Mask downsample ratio
        'dropout': 0.0,  # Dropout regularization
        'val': True,  # Validate during training
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Start training
    print("Starting training...")
    print("=" * 70)
    print()
    
    try:
        results = model.train(**config)
        
        print()
        print("=" * 70)
        print("✅ Training completed successfully!")
        print("=" * 70)
        print()
        print(f"Results saved to: {model.trainer.save_dir}")
        print(f"Best weights: {model.trainer.best}")
        print(f"Last weights: {model.trainer.last}")
        print()
        print("Next steps:")
        print("1. Validate model: yolo segment val model=parking_lot_localization/apklot_stage1/weights/best.pt")
        print("2. Test on wide-area images (zoom 18) to verify parking lot detection")
        print("3. Integrate into multi-stage pipeline for tile planning")
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Training failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
