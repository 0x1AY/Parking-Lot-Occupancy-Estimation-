# Project Structure

## Overview

This document outlines the organization of the Parking Lot Occupancy Estimation project.

## Directory Structure

```
Parking-Lot-Occupancy-Estimation-/
│
├── 📁 occupancy/                          # Main production pipeline
│   ├── unified_parking_pipeline.py        # Complete 4-stage pipeline (dual-model)
│   ├── batch_process.py                   # Batch processing for multiple locations
│   ├── results/                           # Output directory for all results
│   ├── PROJECT_REPORT.md                  # Comprehensive project documentation
│   ├── DUAL_MODEL_IMPLEMENTATION.md       # Dual-model architecture documentation
│   └── README.md                          # Occupancy pipeline usage guide
│
├── 📁 tools/                              # Utility scripts
│   ├── create_multiclass_dataset.py       # Dataset filtering script
│   ├── convert_to_bboxes.py               # Segmentation to bounding box conversion
│   ├── convert_apklot_to_yolo.py          # APKLOT dataset format conversion
│   ├── visualize_apklot.py                # APKLOT visualization tool
│   ├── train_apklot_stage1.py             # Stage 1 localization training
│   ├── plan_tile_coverage.py              # Tile coverage planning utility
│   ├── prepare_upload_dataset.py          # Dataset preparation for upload
│   └── archive/                           # Old/superseded scripts
│
├── 📁 datasets/                           # Model datasets
│   └── apklot/                            # APKLOT parking lot localization dataset
│       ├── apklot_stage1/                 # Stage 1 trained model
│       │   └── weights/best.pt            # Localization model (83.5% mAP50)
│       ├── data.yaml                      # Dataset configuration
│       └── apklot.yaml                    # Training configuration
│
├── 📁 parking_runs/                       # Training outputs
│   ├── yolo11m_parking_augmented2/        # High-accuracy car detection model
│   │   └── weights/best.pt                # Car model (96.3% mAP50, 96.5% recall)
│   └── yolo11m_multilabel/                # Multiclass detection model
│       └── weights/best.pt                # Stall model (84% mAP50)
│
├── 📁 Dataset-V1-detect/                  # Detection dataset (cars + stalls)
│   ├── train/                             # Training split
│   ├── valid/                             # Validation split
│   ├── test/                              # Test split
│   └── data.yaml                          # Dataset configuration
│
├── 📁 Dataset-V1-multiclass/              # Filtered multiclass dataset
│   ├── train/                             # Images with both cars AND stalls
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── 📁 walmart_locations/                  # Walmart location data
│   └── wide_area_z19/                     # Wide-area images (1280x1280, zoom 19)
│
├── 📁 docs/                               # Documentation
│   ├── APKLOT_Dataset_Exploration.md      # Dataset analysis
│   └── APKLOT_Paper_Analysis.md           # Paper review
│
├── 📓 train.ipynb                         # Car detection training notebook
├── 📓 train_multilabel.ipynb              # Multiclass training notebook
├── 📓 validate.ipynb                      # Model validation notebook
├── 📓 visualize.ipynb                     # Dataset visualization notebook
│
├── 📄 README.md                           # Main project README
├── 📄 LICENSE                             # Project license
├── 📄 .gitignore                          # Git ignore rules
├── 📄 Biweekly_Checkin_Report.md          # Progress reports
├── 📄 walmart lots.csv                    # Walmart locations metadata
└── 📄 generate_report.py                  # Report generation script
```

## Key Components

### Production Pipeline (`occupancy/`)

The main production system for parking occupancy detection:

1. **unified_parking_pipeline.py** - Complete 4-stage pipeline:

   - Stage 1: Parking lot localization using APKLOT model
   - Stage 2: High-resolution tile download (zoom 20, 20% overlap)
   - Stage 3: Dual-model object detection (cars + stalls)
   - Stage 4: Tile stitching and occupancy analysis

2. **batch_process.py** - Automated batch processing for multiple locations

3. **results/** - All output files including:
   - Visualizations (overall_occupancy.jpg)
   - Metrics (overall_occupancy.json)
   - Batch summaries (batch_summary.json)

### Models

#### Stage 1: Localization Model

- **Path**: `datasets/apklot/apklot_stage1/weights/best.pt`
- **Architecture**: YOLOv11m-seg
- **Performance**: 83.5% mAP50
- **Purpose**: Detect parking lot boundaries from wide-area imagery

#### Stage 3: Car Detection Model (Dual-Model #1)

- **Path**: `parking_runs/yolo11m_parking_augmented2/weights/best.pt`
- **Architecture**: YOLOv11m
- **Performance**: 96.3% mAP50, 96.5% recall
- **Purpose**: High-accuracy vehicle detection
- **Training**: Specialized car-only detection

#### Stage 3: Stall Detection Model (Dual-Model #2)

- **Path**: `parking_runs/yolo11m_multilabel/weights/best.pt`
- **Architecture**: YOLOv11m
- **Performance**: 84% mAP50
- **Purpose**: Parking stall detection
- **Classes**: Cars, stalls, lot boundaries, objects

### Datasets

#### Dataset-V1-detect

- **Purpose**: Car and stall detection training
- **Format**: YOLO bounding boxes
- **Classes**: car (0), lot_boundary (1), objects (2), stall (3)
- **Images**: 614×614 pixels from BC region

#### Dataset-V1-multiclass

- **Purpose**: Filtered dataset with both cars AND stalls
- **Created by**: `tools/create_multiclass_dataset.py`
- **Training result**: 84% mAP50 validation accuracy

#### APKLOT Dataset

- **Purpose**: Parking lot localization/segmentation
- **Format**: YOLO segmentation masks
- **Usage**: Stage 1 wide-area parking detection

### Training Notebooks

- **train.ipynb** - Car detection model training (96.3% mAP50)
- **train_multilabel.ipynb** - Multiclass model training (84% mAP50)
- **validate.ipynb** - Model performance validation
- **visualize.ipynb** - Dataset and prediction visualization

### Utility Tools

Key scripts in `tools/`:

- **create_multiclass_dataset.py** - Filter dataset to images with both cars and stalls
- **convert_to_bboxes.py** - Convert segmentation masks to bounding boxes
- **convert_apklot_to_yolo.py** - APKLOT format conversion
- **plan_tile_coverage.py** - Calculate optimal tile grid coverage
- **visualize_apklot.py** - Visualize APKLOT annotations

### Documentation

- **README.md** - Main project overview and setup guide
- **occupancy/PROJECT_REPORT.md** - Comprehensive technical report
- **occupancy/DUAL_MODEL_IMPLEMENTATION.md** - Dual-model architecture details
- **docs/** - Dataset exploration and paper analysis

## Usage

### Quick Start

1. **Train car detection model**:

   ```bash
   # Open train.ipynb in Jupyter/Colab
   # Run all cells to train YOLOv11m on car detection
   ```

2. **Run occupancy detection on single location**:

   ```python
   from occupancy.unified_parking_pipeline import UnifiedParkingPipeline

   pipeline = UnifiedParkingPipeline()
   result = pipeline.run_pipeline(
       image_path="walmart_locations/wide_area_z19/location.png",
       center_lat=43.668734,
       center_lon=-79.340158
   )
   ```

3. **Batch process multiple locations**:
   ```bash
   cd occupancy
   python batch_process.py
   ```

### Results

All results are saved to `occupancy/results/` with subdirectories per location containing:

- `overall_occupancy.jpg` - Annotated visualization
- `overall_occupancy.json` - Detailed metrics and data
- `tiles/` - Individual high-resolution tiles

## Data Flow

```
Wide-Area Image (1280x1280, z19)
    ↓
[Stage 1] Localization Model → Parking lot boundaries
    ↓
[Stage 2] Tile Download → Grid of 1280x1280 tiles (z20, 20% overlap)
    ↓
[Stage 3] Dual Detection → Cars (96.3% mAP50) + Stalls (84% mAP50)
    ↓
[Stage 4] Stitching & Analysis → Occupancy metrics + Visualization
    ↓
Output: occupancy.jpg + occupancy.json
```

## Model Performance

| Model           | Purpose           | mAP50     | Recall    | Path                                                    |
| --------------- | ----------------- | --------- | --------- | ------------------------------------------------------- |
| APKLOT Stage 1  | Localization      | 83.5%     | -         | datasets/apklot/apklot_stage1/weights/best.pt           |
| Car Detection   | Vehicle detection | **96.3%** | **96.5%** | parking_runs/yolo11m_parking_augmented2/weights/best.pt |
| Stall Detection | Space detection   | 84.0%     | -         | parking_runs/yolo11m_multilabel/weights/best.pt         |

## Batch Processing Results

Successfully processed 10 Walmart locations:

- **Total stalls**: 813
- **Occupied**: 226 (27.8%)
- **Empty**: 587 (72.2%)
- **Processing success rate**: 100%

See `occupancy/PROJECT_REPORT.md` for detailed results and analysis.

## Development History

1. **Initial multiclass model** - Single model for all classes (84% mAP50)
2. **Tile stitching fix** - Proper 20% overlap handling
3. **Dual-model architecture** - Specialized car model (+14.6% accuracy improvement)
4. **Batch processing** - Automated pipeline for multiple locations

## Notes

- Large model files (.pt) are gitignored
- Dataset images are not included in repository (too large)
- Training outputs (weights/) are gitignored
- All temporary test files are in tools/archive/
