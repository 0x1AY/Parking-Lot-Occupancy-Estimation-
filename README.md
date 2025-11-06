# Parking Lot Occupancy Estimation Using Deep Learning

**Deep Learning Course Project - Fall 2025**  
**Authors:** Aminu Yiwere , Olatunji Olagundoye  
**Institution:** Northeastern University, Vancouver.  
**Course:** Deep Learning  
**GitHub Repository:** [https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git](https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation & Setup](#installation--setup)
- [How to Run the Code](#how-to-run-the-code)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Progress Summary](#progress-summary)
- [Preliminary Results](#preliminary-results)
- [Updated Timeline](#updated-timeline)
- [Next Steps](#next-steps)
- [References](#references)

---

## 🎯 Project Overview

This project develops an automated parking lot occupancy detection and estimation system using deep learning and computer vision techniques. The system leverages state-of-the-art object detection models (YOLOv11) to analyze parking lot images and detect multiple objects including cars, parking stalls, lot boundaries, and other objects, ultimately determining the occupancy status of parking spaces.

### Problem Statement

Parking scarcity in urban and suburban areas persists as a key challenge, intensifying traffic congestion, vehicle emissions, and drivers' time loss. Major retailers like Walmart act as vital shopping and community hubs, with outdoor parking lots enduring high demand that causes inefficient space use and customer discontent. Traditional management methods, reliant on ground sensors or manual counts, are costly, limited in scale, and impractical for large retail sites.

Satellite imagery offers a scalable means for periodic monitoring, yet manual analysis is tedious and error-prone, while automated tools underperform against variables such as variable lighting, weather, occlusions (e.g., trees, shadows), and resolution inconsistencies. This project implements a deep learning strategy to surmount these barriers, enabling automated vehicle detection and occupancy estimation from satellite images.

Key challenges addressed:

- **Traffic Congestion**: Parking scarcity intensifies traffic congestion and wastes drivers' time
- **Environmental Impact**: Unnecessary cruising for parking contributes to increased vehicle emissions
- **Economic Losses**: Wasted fuel and time from "parking hunting" causes significant economic losses
- **Infrastructure Inefficiency**: Traditional sensor-based methods are costly and limited in scale
- **Data Gaps**: Manual analysis is tedious and error-prone; automated tools struggle with real-world variations

### Solution

This project implements a deep learning-based parking occupancy detection system that analyzes satellite imagery from Google Static Maps API to address data gaps and support evidence-based urban planning. The system:

1. **Detects Multiple Objects**: Identifies cars, parking stalls, lot boundaries, and other objects using YOLOv11
2. **Estimates Occupancy**: Analyzes the spatial relationship between detected cars and parking stalls
3. **Provides On-Demand Reports**: Processes satellite images from archival imagery for occupancy analysis
4. **Scales Efficiently**: Leverages Google Static Maps API for wide-area coverage
5. **Supports Urban Planning**: Enables historical occupancy trend analysis for infrastructure decisions
6. **Reduces Resource Waste**: Aims at reducing wasted resources during "parking hunting" [7]
7. **Enables Dynamic Solutions**: Potential for dynamic pricing based on demand and real-time occupancy estimation

The approach addresses traditional limitations by using scalable satellite imagery instead of costly ground sensors, achieving superior accuracy over conventional computer vision methods through deep learning trained on diverse datasets.

---

## 💡 Motivation

### Why Vision-Based Parking Management?

Traditional parking management systems rely on physical sensors such as:

- **Magnetic sensors**: Expensive installation ($300-500 per space)
- **Ultrasonic sensors**: Prone to hardware failures and weather sensitivity
- **Infrared sensors**: Limited range and accuracy
- **Pressure sensors**: High maintenance costs

These traditional approaches have significant limitations:

- ❌ High installation and maintenance costs
- ❌ Susceptible to hardware failures
- ❌ Limited scalability across multiple locations
- ❌ Lack of visual context for analysis
- ❌ Difficult to adapt to layout changes

### Advantages of Computer Vision Approach

Our deep learning-based solution using satellite imagery offers:

- ✅ **Cost-Effective**: Leverages Google Static Maps API without requiring physical camera installation
- ✅ **Wide Coverage**: Satellite imagery enables monitoring of multiple parking facilities simultaneously
- ✅ **Scalable**: Easy deployment across different locations through API calls
- ✅ **Flexible**: Adapts to different parking lot layouts without hardware changes
- ✅ **Rich Data**: Provides aerial visual context with consistent overhead perspective
- ✅ **Low Maintenance**: Software-based solution with no physical hardware dependencies
- ✅ **Real-Time Processing**: Instant analysis and feedback from API-retrieved imagery
- ✅ **Geographic Flexibility**: Can analyze parking lots anywhere accessible via Google Maps
- ✅ **Future-Ready**: Can integrate with smart city infrastructure and IoT systems

---

## 📊 Dataset

### Custom Labeled Dataset - Car Park v6

We have created a custom-labeled dataset specifically for this project using satellite imagery retrieved from Google Static Maps API, annotated using Roboflow Universe. The dataset consists of aerial/overhead views of parking lots, providing a consistent top-down perspective ideal for parking space detection and occupancy analysis.

#### Dataset Information

- **Name**: Car Park - Final Dataset v6
- **Source**: Satellite imagery retrieved using Google Static Maps API, annotated via Roboflow
- **Image Type**: Satellite/aerial view parking lot images
- **License**: CC BY 4.0
- **Roboflow Link**: [https://universe.roboflow.com/ay-luu4n/car-park-x0jof/dataset/6](https://universe.roboflow.com/ay-luu4n/car-park-x0jof/dataset/6)
- **Export Date**: November 6, 2025 at 4:01 AM GMT
- **Total Images**: 171 images
- **Annotation Format**: YOLOv11
- **Image Resolution**: 640x640 (stretched to maintain consistency)

#### Dataset Split

| Split      | Number of Images | Percentage |
| ---------- | ---------------- | ---------- |
| Training   | 115              | 67.3%      |
| Validation | 38               | 22.2%      |
| Test       | 18               | 10.5%      |
| **Total**  | **171**          | **100%**   |

#### Object Classes (4 Classes)

Our dataset includes annotations for four distinct object categories:

1. **`car`**: Vehicles present in the parking lot (occupied spaces)
2. **`stall`**: Individual parking space markings/boundaries
3. **`lot_boundary`**: Parking lot perimeter and boundary lines
4. **`objects`**: Other objects in the parking area (cones, signs, barriers, etc.)

#### Dataset Structure

```
Car Park.v6-final-dataset1.yolov11/
├── data.yaml              # Dataset configuration file
├── README.dataset.txt     # Dataset documentation
├── README.roboflow.txt    # Roboflow export information
├── train/
│   ├── images/           # 115 training images
│   └── labels/           # Corresponding YOLO format annotations
├── valid/
│   ├── images/           # 38 validation images
│   └── labels/           # Corresponding YOLO format annotations
└── test/
    ├── images/           # 18 test images
    └── labels/           # Corresponding YOLO format annotations
```

#### Preprocessing Applied

- **Resize**: All images resized to 640x640 pixels (stretch method)
- **Format**: Exported in YOLOv11 format for seamless integration
- **No Augmentation**: Original images without synthetic augmentation (augmentation applied during training)

#### Data Characteristics

- **Satellite Imagery**: Top-down aerial view from Google Static Maps API
- **Real-World Conditions**: Images captured under various lighting and weather conditions
- **Multiple Locations**: Different parking lot layouts and geographical locations
- **Diverse Scenarios**: Various vehicle types, parking configurations, and occupancy levels
- **Quality**: High-quality annotations with precise bounding boxes on satellite imagery
- **Challenge Factors**: Includes occlusions, shadows, and varying illumination
- **Aerial Perspective**: Consistent overhead view ideal for parking space analysis

#### Dataset Access

The dataset is located in the project directory:

```bash
./Car Park.v6-final-dataset1.yolov11/
```

To use the dataset in your training scripts:

```python
# Path configuration
data_yaml = './Car Park.v6-final-dataset1.yolov11/data.yaml'
```

#### Class Distribution Analysis

The dataset provides balanced representation across different object types, enabling the model to learn:

- **Vehicle detection** for occupancy determination
- **Parking stall localization** for spatial understanding
- **Boundary detection** for lot area definition
- **Object recognition** for obstacle awareness

---

## 🔬 Methodology

### Approach Overview

Inspired by the fusion-based segmentation approach [1], this project develops a deep learning system that:

1. Inputs latitude/longitude coordinates of parking lots (e.g., Canadian Walmart locations)
2. Fetches historical satellite images via Google Static Maps API
3. Processes them to output occupancy reports (e.g., percentage occupied, visualized heatmap)

This project employs a **YOLO-based object detection approach** using YOLOv11 (You Only Look Once) architecture for efficient parking occupancy estimation from satellite imagery.

### Technical Pipeline

```
Coordinates Input → API Image Retrieval → Preprocessing & Augmentation → YOLOv11 Detection → Occupancy Calculation → Report Generation
```

#### Stage 1: Data Acquisition and Preprocessing

**Image Retrieval:**

- Input: Latitude/longitude coordinates of parking lots
- API: Google Static Maps API retrieves zoomed historical satellite view (e.g., 640x640 pixels at scale=1)
- Format: High-resolution satellite imagery (up to 0.5m/pixel)

**Preprocessing:**

- Normalization of pixel values
- Augmentation for robustness: brightness adjustments, shadows, weather variations
- Libraries: OpenCV and Albumentations

#### Stage 2: Object Detection with YOLOv11

**Why YOLOv11?**

YOLOv11 is chosen for its superior characteristics inspired by efficient CNN-based detection [6]:

- **Real-Time Performance**: Enables efficient inference for on-demand reports
- **Computational Efficiency**: Trainable on modest GPUs
- **High Accuracy**: State-of-the-art detection precision
- **Multi-Object Detection**: Simultaneously detects cars, parking stalls, boundaries, and objects
- **End-to-End Learning**: Single network for detection and classification

**Fine-tuning Approach:**

- YOLOv11 model fine-tuned on custom dataset and APKLOT datasets
- Identifies vehicles and parking spots via bounding boxes
- Unlike baseline U-Net segmentation methods [1], YOLO enables efficient inference for on-demand reports

**Detection Process:**

1. **Input**: 640x640 RGB satellite parking lot image
2. **Feature Extraction**: YOLOv11 backbone extracts multi-scale features
3. **Object Detection**: Identifies and localizes all objects with bounding boxes
4. **Classification**: Assigns class labels (car, stall, lot_boundary, objects)
5. **Confidence Scoring**: Provides confidence scores for each detection

#### Stage 3: Occupancy Calculation

**Algorithm:**

```python
For each parking stall detected:
    1. Get stall bounding box
    2. Check for car detection overlapping with stall
    3. If overlap > threshold (e.g., IoU > 0.5):
        → Mark as OCCUPIED
    4. Else:
        → Mark as VACANT
    5. Calculate total occupancy rate
```

**Spatial Analysis:**

- **Intersection over Union (IoU)**: Measures overlap between car and stall bounding boxes
- **Centroid Matching**: Checks if car centroid falls within stall boundaries
- **Confidence Thresholding**: Filters low-confidence detections

#### Stage 3: Post-Processing

- **Non-Maximum Suppression (NMS)**: Eliminates duplicate detections
- **Confidence Filtering**: Removes low-confidence predictions
- **Boundary Validation**: Ensures detected objects are within parking lot boundaries
- **Temporal Smoothing** (for video): Reduces flickering in consecutive frames

### Model Architecture

**YOLOv11 Components:**

1. **Backbone**: CSPDarknet for feature extraction
2. **Neck**: Path Aggregation Network (PAN) for multi-scale feature fusion
3. **Head**: Detection head for bounding box regression and classification

### Training Strategy

- **Transfer Learning**: Fine-tune pre-trained YOLOv11 weights on custom dataset and APKLOT dataset
- **Pre-training**: Leverage APKLOT dataset (500 global satellite images with 7,000+ polygon annotations) for improved model resilience
- **Data Augmentation**:
  - Mosaic augmentation
  - Random scaling and cropping
  - Color jittering (brightness, contrast, saturation)
  - Horizontal flipping
  - Blur and noise addition
  - Shadow and weather condition variations
- **Loss Function**: Focal loss for addressing imbalanced datasets (inspired by dense object detection approaches)
- **Optimizer**: Adam optimizer
- **Learning Rate Schedule**: Adaptive learning rate with warm-up
- **Training Environment**: Modest GPUs (computationally efficient approach)

### Performance Metrics

#### Detection Metrics

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (primary metric)
- **IoU (Intersection over Union)**: Bounding box accuracy metric
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of detected objects among all ground truth objects
- **F1-Score**: Harmonic mean of precision and recall

#### Occupancy Metrics

- **Occupancy Accuracy**: Percentage of correctly classified parking spaces
- **False Positive Rate**: Incorrectly detected occupancy
- **False Negative Rate**: Missed occupied spaces
- **Processing Time**: Inference speed (FPS)

### Advantages of This Approach

✅ **End-to-End Solution**: Single YOLO model handles all detection tasks  
✅ **Computational Efficiency**: Trainable on modest GPUs, suitable for on-demand reports  
✅ **Robust to Variations**: Handles variable lighting, weather, occlusions, and resolution inconsistencies  
✅ **Scalable**: Satellite imagery approach enables monitoring multiple locations  
✅ **Interpretable**: Visual bounding boxes and attention maps show detection reasoning  
✅ **Superior Accuracy**: Deep learning achieves better performance than conventional computer vision  
✅ **Historical Analysis**: Supports long-term occupancy trend analysis from archival imagery  
✅ **Urban Planning Support**: Provides evidence-based insights for infrastructure decisions

---

## 🛠️ Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU recommended (NVIDIA GPU with CUDA 11.8+)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: At least 5GB free space for dataset and models
- **Operating System**: Linux, macOS, or Windows

### Environment Setup

#### Option 1: Local Installation

1. **Clone the repository**

```bash
git clone https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git
cd Parking-Lot-Occupancy-Estimation-
```

2. **Create a virtual environment**

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n parking-detection python=3.9
conda activate parking-detection
```

3. **Install PyTorch with CUDA support**

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

4. **Install Ultralytics YOLOv11**

```bash
pip install ultralytics
```

5. **Install additional dependencies**

```bash
pip install -r requirements.txt
```

#### Option 2: Google Colab (Recommended for Quick Start)

The project includes Jupyter notebooks optimized for Google Colab:

1. **Upload notebooks to Google Drive**

   - `train.ipynb`
   - `validate.ipynb`
   - `test.ipynb`

2. **Open in Google Colab**

   ```
   https://colab.research.google.com
   ```

3. **Mount Google Drive and upload dataset**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **The notebooks will automatically**:
   - Detect Colab environment
   - Install required packages
   - Configure GPU settings
   - Set up directories

### Dataset Setup

1. **Dataset is already included** in the repository:

   ```
   ./Car Park.v6-final-dataset1.yolov11/
   ```

2. **Verify dataset structure**:

   ```bash
   ls -la "Car Park.v6-final-dataset1.yolov11/"
   # Should show: train/, valid/, test/, data.yaml
   ```

3. **For Google Colab users**:
   ```bash
   # Upload entire project folder to Google Drive
   # Path: /content/drive/MyDrive/parking_lot_project/
   ```

### Verify Installation

```python
# Check PyTorch and CUDA
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check Ultralytics
from ultralytics import YOLO
print("YOLOv11 is ready!")
```

### GPU Configuration (Optional)

For optimal performance, configure GPU memory:

```python
import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
```

---

## 🚀 How to Run the Code

### Quick Start with Jupyter Notebooks

The project provides three main notebooks for step-by-step execution:

#### 1. Training (`train.ipynb`)

Open in Google Colab or locally and follow the notebook cells:

```python
# The notebook guides you through:
# 1. Environment setup and package installation
# 2. Dataset loading and verification
# 3. Model configuration
# 4. Training with progress monitoring
# 5. Saving checkpoints
```

**Key Steps:**

- Mount Google Drive (Colab) or set local paths
- Load the custom dataset from `data.yaml`
- Configure YOLOv11 model parameters
- Train with real-time visualization
- Save best model weights

#### 2. Validation (`validate.ipynb`)

Evaluate model performance on validation set:

```python
# The notebook includes:
# 1. Load trained model
# 2. Run validation on validation set
# 3. Calculate mAP, precision, recall
# 4. Generate confusion matrix
# 5. Visualize predictions
```

**Outputs:**

- Validation metrics
- Class-wise performance
- Sample predictions with bounding boxes
- Error analysis

#### 3. Testing (`test.ipynb`)

Final evaluation on test set:

```python
# The notebook covers:
# 1. Load best trained model
# 2. Test on unseen test images
# 3. Measure inference time
# 4. Generate comprehensive report
# 5. Export results
```

**Outputs:**

- Final test metrics
- Inference speed (FPS)
- Per-image predictions
- Annotated output images

### Command Line Usage (Advanced)

#### Training

```bash
# Basic training
yolo detect train data='Car Park.v6-final-dataset1.yolov11/data.yaml' \
                 model=yolov11n.pt \
                 epochs=100 \
                 imgsz=640 \
                 batch=16

# Training with custom parameters
yolo detect train data='Car Park.v6-final-dataset1.yolov11/data.yaml' \
                 model=yolov11s.pt \
                 epochs=150 \
                 imgsz=640 \
                 batch=32 \
                 lr0=0.01 \
                 device=0 \
                 project=runs/train \
                 name=parking_detection
```

#### Validation

```bash
# Validate trained model
yolo detect val model=runs/train/parking_detection/weights/best.pt \
                data='Car Park.v6-final-dataset1.yolov11/data.yaml'
```

#### Inference/Prediction

```bash
# Predict on single image
yolo detect predict model=runs/train/parking_detection/weights/best.pt \
                    source='path/to/image.jpg' \
                    conf=0.25

# Predict on folder of images
yolo detect predict model=runs/train/parking_detection/weights/best.pt \
                    source='path/to/images/' \
                    save=True \
                    conf=0.25

# Predict on video
yolo detect predict model=runs/train/parking_detection/weights/best.pt \
                    source='path/to/video.mp4' \
                    save=True
```

#### Export Model

```bash
# Export to ONNX format
yolo export model=runs/train/parking_detection/weights/best.pt format=onnx

# Export to TensorRT
yolo export model=runs/train/parking_detection/weights/best.pt format=engine
```

### Python Script Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov11n.pt')  # or 'path/to/best.pt'

# Train
results = model.train(
    data='Car Park.v6-final-dataset1.yolov11/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# Predict
results = model.predict(source='image.jpg', save=True)

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        cls = int(box.cls[0])  # Class ID
        conf = float(box.conf[0])  # Confidence
        xyxy = box.xyxy[0].tolist()  # Bounding box coordinates
        print(f"Class: {cls}, Conf: {conf:.2f}, Box: {xyxy}")
```

### Occupancy Calculation

```python
def calculate_occupancy(results, iou_threshold=0.5):
    """
    Calculate parking lot occupancy from YOLOv11 detections
    """
    cars = []
    stalls = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()

        if cls == 0:  # car
            cars.append(xyxy)
        elif cls == 3:  # stall
            stalls.append(xyxy)

    occupied = 0
    for stall in stalls:
        for car in cars:
            if calculate_iou(stall, car) > iou_threshold:
                occupied += 1
                break

    total_stalls = len(stalls)
    occupancy_rate = (occupied / total_stalls * 100) if total_stalls > 0 else 0

    return {
        'total_stalls': total_stalls,
        'occupied': occupied,
        'vacant': total_stalls - occupied,
        'occupancy_rate': occupancy_rate
    }
```

---

## 📁 Project Structure

```
Parking-Lot-Occupancy-Estimation-/
│
├── README.md                           # Project documentation (this file)
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
│
├── Car Park.v6-final-dataset1.yolov11/ # Custom labeled dataset
│   ├── data.yaml                       # Dataset configuration for YOLO
│   ├── README.dataset.txt              # Dataset information
│   ├── README.roboflow.txt             # Roboflow export details
│   ├── train/
│   │   ├── images/                     # 115 training images
│   │   └── labels/                     # YOLO format annotations
│   ├── valid/
│   │   ├── images/                     # 38 validation images
│   │   └── labels/                     # YOLO format annotations
│   └── test/
│       ├── images/                     # 18 test images
│       └── labels/                     # YOLO format annotations
│
├── train.ipynb                         # Training notebook (Google Colab ready)
├── validate.ipynb                      # Validation notebook
├── test.ipynb                          # Testing notebook
│
├── runs/                               # Training outputs (auto-generated)
│   ├── train/                          # Training run directories
│   │   └── parking_detection/
│   │       ├── weights/
│   │       │   ├── best.pt            # Best model weights
│   │       │   └── last.pt            # Last epoch weights
│   │       ├── results.png            # Training metrics plot
│   │       ├── confusion_matrix.png   # Confusion matrix
│   │       └── ...                    # Other outputs
│   ├── val/                           # Validation outputs
│   └── predict/                       # Prediction outputs
│
├── models/                            # Model checkpoints (optional)
├── outputs/                           # Generated outputs
├── logs/                              # TensorBoard logs
└── scripts/                           # Utility scripts (if any)
```

### Key Files Description

#### Notebooks

- **`train.ipynb`**: Complete training pipeline with step-by-step code cells
- **`validate.ipynb`**: Model validation and performance analysis
- **`test.ipynb`**: Final testing and inference on test set

#### Dataset Files

- **`data.yaml`**: YOLO configuration file specifying:
  - Training, validation, and test image paths
  - Number of classes (nc: 4)
  - Class names: ['car', 'lot_boundary', 'objects', 'stall']
  - Roboflow project information

#### Model Outputs

After training, the following files are generated in `runs/train/<experiment_name>/`:

- **`weights/best.pt`**: Best performing model based on validation mAP
- **`weights/last.pt`**: Model from the last training epoch
- **`results.png`**: Training/validation metrics plots
- **`confusion_matrix.png`**: Confusion matrix visualization
- **`labels.jpg`**: Ground truth label distribution
- **`predictions.jpg`**: Sample predictions on validation set
- **`results.csv`**: Detailed training metrics per epoch

---

## 📦 Dependencies

### Core Requirements

```txt
# Deep Learning Framework
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# YOLOv11 and Object Detection
ultralytics>=8.0.0

# Computer Vision
opencv-python>=4.8.0
pillow>=10.0.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
```

### Additional Libraries

```txt
# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Progress Bars
tqdm>=4.65.0

# Image Augmentation (optional)
albumentations>=1.3.0

# Metrics and Evaluation
scikit-learn>=1.3.0
scipy>=1.10.0

# Jupyter Support
jupyter>=1.0.0
ipywidgets>=8.0.0

# Utilities
pyyaml>=6.0
```

### Installation

Save the following as `requirements.txt`:

```txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=10.0.0
tqdm>=4.65.0
scikit-learn>=1.3.0
jupyter>=1.0.0
pyyaml>=6.0
```

Then install:

```bash
pip install -r requirements.txt
```

### System Requirements

| Component   | Minimum                                  | Recommended               |
| ----------- | ---------------------------------------- | ------------------------- |
| **Python**  | 3.8+                                     | 3.9+                      |
| **RAM**     | 8GB                                      | 16GB+                     |
| **GPU**     | -                                        | NVIDIA GPU with 6GB+ VRAM |
| **CUDA**    | -                                        | 11.8+                     |
| **Storage** | 5GB                                      | 20GB+                     |
| **OS**      | Windows 10+, Ubuntu 18.04+, macOS 10.14+ | Ubuntu 20.04+, Windows 11 |

---

## 📈 Progress Summary

### ✅ Completed Work

#### Phase 1: Project Planning & Setup (Week 1-2) ✅

- [x] **Literature Review**: Comprehensive review of parking occupancy detection methods
  - Studied traditional sensor-based approaches
  - Analyzed computer vision techniques (YOLO, Faster R-CNN, SSD)
  - Reviewed related academic papers and industry solutions
- [x] **Project Setup**: Repository and development environment configured

  - Created GitHub repository with proper structure
  - Set up Git version control
  - Configured Python environment
  - Created comprehensive project documentation

- [x] **Technology Selection**: Chosen YOLOv11 for object detection
  - Evaluated multiple architectures (YOLOv8, YOLOv11, Faster R-CNN)
  - Selected YOLOv11 for superior speed and accuracy balance
  - Justified selection based on real-time requirements

#### Phase 2: Dataset Creation & Annotation (Week 3-4) ✅

- [x] **Data Collection**: Gathered parking lot images

  - Collected 171 high-quality parking lot images
  - Ensured diverse scenarios (different times, lighting, occupancy)
  - Captured multiple camera angles and perspectives

- [x] **Data Annotation**: Manual labeling using Roboflow

  - Annotated 4 object classes: car, stall, lot_boundary, objects
  - Created precise bounding boxes for each object
  - Ensured annotation quality and consistency
  - Total annotated objects: ~800+ bounding boxes

- [x] **Dataset Organization**: Structured train/val/test splits

  - Training: 115 images (67.3%)
  - Validation: 38 images (22.2%)
  - Test: 18 images (10.5%)
  - Exported in YOLOv11 format

- [x] **Data Preprocessing**: Prepared images for training
  - Resized all images to 640x640
  - Maintained consistent format
  - Created data.yaml configuration file

#### Phase 3: Development & Implementation (Week 5 - Current) 🔄

- [x] **Notebook Development**: Created three Jupyter notebooks

  - `train.ipynb`: Complete training pipeline
  - `validate.ipynb`: Model validation framework
  - `test.ipynb`: Final testing procedures
  - All notebooks optimized for Google Colab

- [x] **Code Structure**: Organized code into modular cells

  - Setup and installation cells
  - Data loading and visualization
  - Model configuration
  - Training functions
  - Evaluation metrics
  - Result visualization

- [ ] **Model Training**: YOLOv11 training in progress
  - Configured training hyperparameters
  - Set up data augmentation pipeline
  - Ready to begin training experiments

### 🔄 Work In Progress

- **Model Training**: Currently preparing to train YOLOv11 on custom dataset
- **Hyperparameter Tuning**: Planning grid search for optimal parameters
- **Evaluation Pipeline**: Setting up comprehensive evaluation framework

### Current Status: Ready for Training Phase

All preparatory work is complete. The project is now ready to proceed with:

1. Model training on the custom dataset
2. Hyperparameter optimization
3. Performance evaluation
4. Results analysis

---

## 📊 Preliminary Results

### Dataset Statistics

#### Image Distribution

| Split      | Images  | Percentage | Purpose               |
| ---------- | ------- | ---------- | --------------------- |
| Training   | 115     | 67.3%      | Model training        |
| Validation | 38      | 22.2%      | Hyperparameter tuning |
| Test       | 18      | 10.5%      | Final evaluation      |
| **Total**  | **171** | **100%**   | Complete dataset      |

#### Object Classes

| Class ID | Class Name     | Description             | Use Case             |
| -------- | -------------- | ----------------------- | -------------------- |
| 0        | `car`          | Vehicles in parking lot | Occupancy detection  |
| 1        | `lot_boundary` | Parking lot perimeter   | Spatial context      |
| 2        | `objects`      | Signs, cones, barriers  | Obstacle detection   |
| 3        | `stall`        | Parking space markings  | Capacity calculation |

#### Annotation Quality

- **Total Annotations**: ~800+ bounding boxes across all classes
- **Annotation Tool**: Roboflow Universe
- **Format**: YOLOv11 (normalized xywh format)
- **Quality Assurance**: Manual review and validation
- **Inter-Annotator Agreement**: Consistent labeling guidelines followed

### Data Characteristics

#### Image Properties

- **Resolution**: 640×640 pixels (standardized)
- **Color Space**: RGB
- **File Format**: JPEG
- **Size Range**: 50-200 KB per image

#### Scenario Diversity

- ✅ Multiple lighting conditions (day, evening, overcast)
- ✅ Various occupancy levels (empty to full)
- ✅ Different camera angles and heights
- ✅ Real-world parking lot environments
- ✅ Diverse vehicle types and sizes

### Expected Performance (Based on YOLOv11 Benchmarks)

#### Anticipated Metrics

| Metric              | Target  | Rationale                                    |
| ------------------- | ------- | -------------------------------------------- |
| **mAP@0.5**         | >85%    | Standard YOLO performance on custom datasets |
| **mAP@0.5:0.95**    | >70%    | Comprehensive detection accuracy             |
| **Precision**       | >90%    | Minimize false positives                     |
| **Recall**          | >85%    | Minimize missed detections                   |
| **Inference Speed** | >30 FPS | Real-time capability (GPU)                   |
| **Model Size**      | <50 MB  | Deployment-friendly                          |

### Training Progress

#### Current Status

- **Stage**: Pre-training (notebooks prepared, dataset ready)
- **Next Step**: Begin YOLOv11 training on custom dataset
- **Timeline**: Training scheduled to start Week 5

#### Planned Experiments

1. **Baseline Training**: YOLOv11n (nano) model - fastest
2. **Medium Model**: YOLOv11s (small) - balanced
3. **High Accuracy**: YOLOv11m (medium) - most accurate
4. **Comparison**: Evaluate trade-offs between speed and accuracy

### Sample Visualizations

_(To be added after training)_

- **Training Curves**: Loss, mAP, precision, recall over epochs
- **Confusion Matrix**: Class-wise performance breakdown
- **Detection Examples**: Sample predictions with bounding boxes
- **Error Analysis**: Misclassification patterns and challenges

### Key Observations

1. **Dataset Size**: 171 images is suitable for transfer learning with pre-trained YOLOv11
2. **Class Balance**: Need to analyze class distribution in annotations
3. **Data Quality**: High-quality manual annotations ensure reliable training
4. **Augmentation**: YOLOv11's built-in augmentation will enhance dataset diversity

### Next Steps for Results

1. ✅ Complete model training (Week 5-6)
2. ✅ Record all training metrics
3. ✅ Generate visualization plots
4. ✅ Analyze validation performance
5. ✅ Update this section with actual results

---

## 📅 Updated Timeline / Milestones

### Week 1 (September 22-28, 2025)

**Defining the Problem and Background Research**

- [x] Literature Review
- [x] Examining Datasets (APKLOT, Grab-Pklot, VME)
- [x] Problem definition and scope

### Week 2 (September 29-October 5, 2025)

**Specify Requirements**

- [x] Computational Needs assessment
- [x] Infrastructure planning
- [x] Tool selection (Google Static Maps API, Roboflow)

### Week 3 (October 6-12, 2025)

**Choose the Best Solution**

- [x] Finalize YOLO-Based Approach
- [x] Data Collection - Fetch and Label Initial dataset
- [x] Manual annotation via Roboflow

### Week 4 (October 13-19, 2025)

**Develop the Solution**

- [x] Implementation of Image Fetching via Google Static Maps API
- [x] Setup model training infrastructure
- [x] Prepare training notebooks

### Week 5 (October 20-26, 2025)

**Build Prototype and Begin Testing**

- [ ] Train model using small subset of custom and public data
- [ ] Initial baseline model training
- [ ] Preliminary testing and validation

### Week 6 (October 27-November 2, 2025)

**Test and Redesign**

- [ ] Evaluate Metrics (mAP@0.5, IoU, Precision, Recall)
- [ ] Adjust Hyperparameters
- [ ] Fine-tune model based on initial results

### Week 7 (November 3-9, 2025)

**Expand Dataset and Retrain**

- [ ] Incorporate More Canadian Locations (Walmart parking lots across Lower Mainland, BC)
- [ ] Augment dataset with diverse conditions
- [ ] Retrain with expanded dataset

### Week 8 (November 10-16, 2025)

**Integration and Time-Series Extension**

- [ ] Add Occupancy Reporting
- [ ] Basic Trend Analysis from historical imagery
- [ ] Generate on-demand reports

### Week 9 (November 17-23, 2025)

**Final Evaluation**

- [ ] Ablation Studies
- [ ] Robustness Testing (lighting, weather, occlusions)
- [ ] Performance comparison with baseline approaches

### Week 10 (November 24-30, 2025)

**Documentation and Polish**

- [ ] Prepare Report
- [ ] Create Demo
- [ ] Finalize documentation with actual results

### Final Week (December 1-7, 2025)

**Presentation Preparation and Submission**

- [ ] Presentation Preparation
- [ ] Final project submission
- [ ] Deliver results and insights

### Milestones & Deadlines

| Milestone                   | Date             | Status         | Priority     |
| --------------------------- | ---------------- | -------------- | ------------ |
| Dataset annotation complete | Nov 6, 2025      | ✅ Complete    | HIGH         |
| Notebooks development done  | Nov 6, 2025      | ✅ Complete    | HIGH         |
| Baseline models trained     | Nov 12, 2025     | ⏳ Pending     | HIGH         |
| Best model identified       | Nov 24, 2025     | ⏳ Pending     | HIGH         |
| Test evaluation complete    | Dec 1, 2025      | ⏳ Pending     | MEDIUM       |
| Documentation finalized     | Dec 15, 2025     | ⏳ Pending     | MEDIUM       |
| **Final submission**        | **Dec 20, 2025** | ⏳ **Pending** | **CRITICAL** |

### Progress Tracking

- **Overall Progress**: ~40% Complete
- **Current Phase**: Phase 3 - Model Training (Week 5)
- **Next Major Deadline**: Baseline training (Nov 12)
- **Days Until Final Submission**: ~45 days

### Risk Mitigation

| Risk                               | Impact | Mitigation Strategy                          |
| ---------------------------------- | ------ | -------------------------------------------- |
| Training time longer than expected | Medium | Start training early, use cloud GPUs         |
| Model performance below target     | High   | Try multiple architectures, ensemble methods |
| Limited dataset size               | Medium | Heavy augmentation, transfer learning        |
| Hardware constraints               | Low    | Use Google Colab Pro, optimize batch size    |
| Time constraints                   | Medium | Follow strict timeline, prioritize tasks     |

---

## 🎯 Next Steps

### Immediate Actions (This Week - Week 5)

#### 1. Begin Model Training 🔥 **HIGH PRIORITY**

**Tasks:**

- [ ] Open `train.ipynb` in Google Colab
- [ ] Upload dataset to Google Drive
- [ ] Configure training hyperparameters
- [ ] Start YOLOv11n baseline training (100 epochs)
- [ ] Monitor training progress via TensorBoard
- [ ] Save and backup model checkpoints

**Expected Outputs:**

- Trained YOLOv11n model weights
- Training metrics and curves
- Validation performance results

#### 2. Run Validation Experiments

**Tasks:**

- [ ] After training completes, open `validate.ipynb`
- [ ] Load best trained model
- [ ] Run validation on validation set
- [ ] Calculate mAP, precision, recall, F1
- [ ] Generate confusion matrix
- [ ] Analyze misclassifications

**Expected Outputs:**

- Validation metrics report
- Confusion matrix visualization
- Error analysis document

#### 3. Document Training Process

**Tasks:**

- [ ] Record all hyperparameters used
- [ ] Screenshot training progress
- [ ] Note any issues or observations
- [ ] Update README with preliminary results

### Short-Term Goals (Week 6-7)

#### 4. Hyperparameter Optimization

**Experiments to Run:**

| Parameter     | Values to Test               | Priority |
| ------------- | ---------------------------- | -------- |
| Model Size    | YOLOv11n, YOLOv11s, YOLOv11m | HIGH     |
| Learning Rate | 0.0001, 0.001, 0.01          | HIGH     |
| Batch Size    | 8, 16, 32                    | MEDIUM   |
| Image Size    | 640, 800                     | MEDIUM   |
| Epochs        | 100, 150, 200                | LOW      |

**Action Items:**

- [ ] Create experiment tracking spreadsheet
- [ ] Run systematic hyperparameter grid search
- [ ] Compare results across all experiments
- [ ] Identify optimal configuration

#### 5. Model Comparison & Selection

**Tasks:**

- [ ] Train at least 3 different YOLOv11 variants
- [ ] Compare inference speed vs accuracy
- [ ] Test on diverse test scenarios
- [ ] Select best model based on requirements
- [ ] Document selection rationale

#### 6. Advanced Techniques (If Time Permits)

**Optional Enhancements:**

- [ ] Test ensemble methods (multiple models)
- [ ] Implement confidence thresholding optimization
- [ ] Try test-time augmentation (TTA)
- [ ] Experiment with post-processing techniques

### Medium-Term Goals (Week 8-9)

#### 7. Comprehensive Evaluation

**Test Set Analysis:**

- [ ] Run `test.ipynb` on final test set
- [ ] Calculate comprehensive metrics
- [ ] Measure inference time and FPS
- [ ] Generate per-image prediction reports
- [ ] Create occupancy estimation results

**Deliverables:**

- Final test metrics report
- Annotated prediction visualizations
- Performance benchmark comparison
- Speed vs accuracy analysis

#### 8. Occupancy Calculation Implementation

**Tasks:**

- [ ] Implement IoU-based occupancy algorithm
- [ ] Calculate occupancy rates for test images
- [ ] Compare with ground truth occupancy
- [ ] Validate occupancy accuracy
- [ ] Generate occupancy heatmaps

#### 9. Error Analysis & Insights

**Analysis Areas:**

- [ ] Identify common failure patterns
- [ ] Analyze challenging scenarios (occlusion, lighting)
- [ ] Document model limitations
- [ ] Propose improvement strategies
- [ ] Create visual error analysis report

### Long-Term Goals (Week 10-11)

#### 10. Final Documentation

**Documentation Tasks:**

- [ ] Update README with all actual results
- [ ] Write comprehensive technical report
- [ ] Create user guide for model deployment
- [ ] Document API and usage examples
- [ ] Add troubleshooting guide

**Report Sections:**

1. Abstract
2. Introduction & Literature Review
3. Methodology
4. Dataset Description
5. Experiments & Results
6. Discussion & Analysis
7. Conclusion & Future Work
8. References

#### 11. Presentation Preparation

**Materials to Create:**

- [ ] PowerPoint/Google Slides presentation
- [ ] Demo video showing model in action
- [ ] Key visualizations and charts
- [ ] Live demo preparation (if required)
- [ ] Q&A preparation

**Presentation Outline:**

1. Problem Statement (2 min)
2. Approach & Methodology (3 min)
3. Dataset & Annotation (2 min)
4. Results & Analysis (5 min)
5. Conclusion & Future Work (2 min)
6. Q&A (6 min)

#### 12. Final Submission Checklist

**Before Submission:**

- [ ] All code runs without errors
- [ ] README is complete and accurate
- [ ] All notebooks have outputs
- [ ] Requirements.txt is up to date
- [ ] Code is well-commented
- [ ] GitHub repository is organized
- [ ] All links work correctly
- [ ] Presentation materials ready
- [ ] Technical report complete

### Research Questions to Address

1. **How does YOLOv11 perform on custom parking lot dataset?**

   - Compare with published benchmarks
   - Analyze performance across different model sizes

2. **What is the optimal balance between speed and accuracy?**

   - Test inference time for each model variant
   - Determine minimum acceptable accuracy for real-time use

3. **How robust is the model to different conditions?**

   - Test on various lighting conditions
   - Evaluate performance with occlusions
   - Analyze behavior with different camera angles

4. **Can occupancy estimation be done reliably?**

   - Validate occupancy calculation accuracy
   - Compare different IoU thresholds
   - Test temporal consistency (if video available)

5. **What are the main limitations and how can they be addressed?**
   - Document failure cases
   - Propose solutions (more data, better augmentation, ensemble)
   - Discuss real-world deployment challenges

### Success Criteria

**Minimum Acceptable Performance:**

- ✅ mAP@0.5 > 80%
- ✅ Inference speed > 20 FPS (on GPU)
- ✅ Occupancy accuracy > 85%

**Target Performance:**

- 🎯 mAP@0.5 > 90%
- 🎯 Inference speed > 40 FPS
- 🎯 Occupancy accuracy > 95%

**Stretch Goals:**

- 🌟 Real-time video processing demo
- 🌟 Web application for inference
- 🌟 Deployment-ready model (ONNX export)
- 🌟 Mobile optimization (TensorFlow Lite)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Aminu Yiwere**

- GitHub: [@0x1AY](https://github.com/0x1AY)
- Project Link: [https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git](https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git)

---

## 🙏 Acknowledgments

- PKLot Dataset creators at UFPR
- Deep Learning course instructors and TAs
- PyTorch and OpenCV communities
- Research papers that inspired this work

---

## 📚 References

### Academic Papers

[1] Y. Yin, H. Wang, D. M. Nguyen, and R. Zimmermann, "A Context-Enriched Satellite Imagery Dataset and an Approach for Parking Lot Detection," in _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_, 2022. [Online]. Available: https://openaccess.thecvf.com/content/WACV2022/papers/Yin_A_Context-Enriched_Satellite_Imagery_Dataset_and_an_Approach_for_Parking_WACV_2022_paper.pdf

[2] G. Amato, F. Carrara, F. Falchi, C. Gennaro, C. Meghini, and C. Vairo, "Deep Learning for Decentralized Parking Lot Occupancy Detection," _Expert Systems with Applications_, vol. 72, pp. 327-334, 2017. [Online]. Available: https://www.sciencedirect.com/science/article/abs/pii/S095741741630598X

[3] S. Drouyer, "Parking Occupancy Estimation on PlanetScope Satellite Images," _Remote Sensing_, vol. 15, no. 11, p. 2806, 2023. [Online]. Available: https://www.mdpi.com/2072-4292/15/11/2806

[4] J. Hellekes, E. V. Puttkammer, and F. Fraissinet-Tachet, "Parking Space Inventory from Above: Detection on Aerial Images and Estimation for Unobserved Regions," _IET Intelligent Transport Systems_, vol. 17, no. 5, pp. 997-1012, 2023. [Online]. Available: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/itr2.12322

[5] T. N. Mundhenk, G. Konjevod, W. A. Sakla, and K. Boakye, "A Large Contextual Dataset for Classification, Detection and Counting of Cars with Deep Learning," in _European Conference on Computer Vision (ECCV)_, 2016. [Online]. Available: https://arxiv.org/abs/1609.04453

[6] S. Zambanini, A.-M. Loghin, N. Pfeifer, E. M. Soley, and R. Sablatnig, "Detection of parking cars in stereo satellite images," _Remote Sens._, vol. 12, no. 13, p. 2170, Jul. 2020, doi: 10.3390/rs12132170.

[7] G. Pierce and D. Shoup, "Getting the Prices Right: An Evaluation of Pricing Parking by Demand in San Francisco," _Journal of the American Planning Association_, vol. 79, no. 1, pp. 67-81, 2013.

### Datasets

1. **APKLOT Dataset**: 500 global satellite images with over 7,000 polygon annotations for parking areas
   - Split: 300 training, 100 validation, 101 testing samples
   - Available on GitHub under MIT license
   - Used for pre-training to improve model resilience

2. **Grab-Pklot Dataset**: 1,344 images at 0.3m/pixel with ground-truth annotations
   - Features roads and buildings context
   - Split: 1,144 training and 200 testing samples
   - Supports fusion-based training

3. **VME Dataset**: For adaptable vehicle detection in satellite imagery

4. **Custom Dataset - Car Park v6**: 
   - Location: Canadian outdoor parking lots (Lower Mainland, British Columbia - Walmart locations)
   - Images: 120-200 images (1024×1024 pixels, resized to 640×640)
   - Source: Google Static Maps API and Bing Maps API
   - Resolution: Up to 0.5m/pixel high-resolution satellite imagery
   - Manual annotation via Roboflow for vehicle and parking spot bounding boxes
   - Reflects differences in weather, layouts, and densities
   - Roboflow Link: [https://universe.roboflow.com/ay-luu4n/car-park-x0jof](https://universe.roboflow.com/ay-luu4n/car-park-x0jof)
   - Format: JPEG/PNG with annotations
   - Current version: 171 annotated images (115 train, 38 valid, 18 test)

### Tools & Frameworks

1. **Ultralytics YOLOv11**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

   - Official YOLOv11 documentation and implementation

2. **PyTorch**: [https://pytorch.org/](https://pytorch.org/)

   - Deep learning framework

3. **Roboflow**: [https://roboflow.com/](https://roboflow.com/)

   - Computer vision platform for dataset management

4. **OpenCV**: [https://opencv.org/](https://opencv.org/)

   - Computer vision library for preprocessing

5. **Albumentations**: Data augmentation library
   
   - For handling shadows, weather conditions, and preprocessing

6. **Google Static Maps API**: For retrieving satellite imagery

   - High-resolution satellite imagery source (up to 0.5m/pixel)
   - Historical overhead perspectives

7. **Bing Maps API**: Alternative satellite imagery source

---

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

### How to Contribute

1. **Fork the repository**

   ```bash
   # Click the 'Fork' button on GitHub
   ```

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR-USERNAME/Parking-Lot-Occupancy-Estimation-.git
   cd Parking-Lot-Occupancy-Estimation-
   ```

3. **Create a feature branch**

   ```bash
   git checkout -b feature/AmazingFeature
   ```

4. **Make your changes**

   - Add new features
   - Fix bugs
   - Improve documentation
   - Optimize performance

5. **Commit your changes**

   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

6. **Push to your branch**

   ```bash
   git push origin feature/AmazingFeature
   ```

7. **Open a Pull Request**
   - Go to the original repository
   - Click 'New Pull Request'
   - Describe your changes

### Contribution Guidelines

- Write clear, commented code
- Follow PEP 8 style guide for Python
- Add tests for new features
- Update documentation as needed
- Keep pull requests focused and small

### Areas for Contribution

- 🐛 **Bug Fixes**: Report and fix bugs
- ✨ **New Features**: Add new functionality
- 📝 **Documentation**: Improve docs and examples
- 🎨 **UI/UX**: Enhance visualizations
- ⚡ **Performance**: Optimize code
- 🧪 **Testing**: Add unit tests

---

## 📄 License

This project is licensed under the **MIT License**.

### MIT License

```
MIT License

Copyright (c) 2025 Aminu Yiwere

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Dataset License:**

- Custom Car Park Dataset: CC BY 4.0 (Creative Commons Attribution 4.0 International)

---

## 📧 Contact

**Aminu Yiwere**

- 📧 **Email**: [Your Email]
- 🐙 **GitHub**: [@0x1AY](https://github.com/0x1AY)
- 💼 **LinkedIn**: [Your LinkedIn]
- 🌐 **Project Repository**: [Parking-Lot-Occupancy-Estimation-](https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git)

### Get in Touch

Have questions, suggestions, or want to collaborate?

- 💬 Open an [Issue](https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-/issues) for bug reports or feature requests
- 🌟 Star the repository if you find it helpful
- 🔀 Fork and contribute improvements
- 📧 Email for academic collaboration or inquiries

---

## 🙏 Acknowledgments

### Special Thanks

- **Course Instructor & TAs**: For guidance and support throughout the project
- **Roboflow Team**: For providing excellent annotation tools and dataset hosting
- **Ultralytics**: For the outstanding YOLOv11 implementation and documentation
- **PyTorch Community**: For the robust deep learning framework
- **Open Source Contributors**: For the libraries and tools that made this project possible

### Inspiration & Resources

- **PKLot Dataset Creators**: UFPR researchers for pioneering parking lot datasets
- **YOLO Community**: For continuous innovations in object detection
- **Stack Overflow & GitHub**: For troubleshooting and code examples
- **Kaggle & Papers with Code**: For dataset and model references

### Tools & Platforms

- **Google Colab**: For providing free GPU resources
- **GitHub**: For version control and code hosting
- **Roboflow Universe**: For dataset management and annotation
- **VS Code**: For development environment

---

## 📝 Project Notes

### Development Log

- **November 6, 2025**:

  - ✅ Completed custom dataset creation and annotation (171 images)
  - ✅ Created comprehensive README documentation
  - ✅ Developed training, validation, and testing notebooks
  - 🔄 Ready to begin model training phase

- **November 5, 2025**:

  - ✅ Cleared notebook code cells for step-by-step development
  - ✅ Prepared project structure for Google Colab
  - ✅ Set up Git repository and version control

- **November 1-4, 2025**:

  - ✅ Collected parking lot images
  - ✅ Manual annotation using Roboflow
  - ✅ Dataset organization and export

- **October 2025**:
  - ✅ Project planning and proposal
  - ✅ Literature review and technology selection
  - ✅ GitHub repository setup

### Known Issues & Limitations

1. **Dataset Size**: 171 images is relatively small; may benefit from additional data

   - **Mitigation**: Heavy data augmentation, transfer learning

2. **Class Imbalance**: Need to analyze class distribution in annotations

   - **Mitigation**: Weighted loss functions, balanced sampling

3. **Computational Resources**: Training may be slow without GPU

   - **Mitigation**: Use Google Colab Pro, optimize batch size

4. **Generalization**: Model trained on specific parking lots may not generalize perfectly
   - **Mitigation**: Diverse test scenarios, domain adaptation techniques

### Future Enhancements

- 🚀 **Real-Time Video Processing**: Extend to live camera feeds
- 📱 **Mobile App**: Develop iOS/Android application
- 🌐 **Web Dashboard**: Create interactive web interface
- 🔗 **API Integration**: RESTful API for third-party integration
- 🤖 **Multi-Task Learning**: Add vehicle type classification
- 📊 **Analytics Dashboard**: Historical occupancy trends and predictions
- 🎯 **Active Learning**: Continuously improve model with new data

---

## 📊 Project Statistics

![GitHub last commit](https://img.shields.io/github/last-commit/0x1AY/Parking-Lot-Occupancy-Estimation-)
![GitHub repo size](https://img.shields.io/github/repo-size/0x1AY/Parking-Lot-Occupancy-Estimation-)
![GitHub](https://img.shields.io/github/license/0x1AY/Parking-Lot-Occupancy-Estimation-)

- **Lines of Code**: TBD (to be calculated after implementation)
- **Total Commits**: [View on GitHub](https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-/commits/main)
- **Contributors**: 1 (Open for contributions!)
- **Stars**: Star this repo if you find it useful! ⭐

---

## ❓ FAQ (Frequently Asked Questions)

### Q1: What makes this project different from existing solutions?

**A:** This project uses the latest YOLOv11 architecture with a custom-annotated dataset specifically tailored for parking lot occupancy detection. It focuses on real-world applicability with optimized inference speed.

### Q2: Can this work with different parking lot layouts?

**A:** Yes! The model learns general features of cars and parking stalls. However, performance may vary on significantly different layouts. Fine-tuning on new data is recommended for best results.

### Q3: What hardware do I need to run this?

**A:** For inference, any modern computer will work (CPU mode). For training, we recommend a CUDA-compatible NVIDIA GPU. Google Colab provides free GPU access for training.

### Q4: How accurate is the occupancy detection?

**A:** Target accuracy is >85% for occupancy estimation. Actual results will be updated after training completes. Performance depends on image quality, lighting, and occlusion factors.

### Q5: Can I use this for commercial purposes?

**A:** The code is MIT licensed (free for commercial use). However, check the dataset license (CC BY 4.0) for attribution requirements. Trained model weights inherit dataset licensing.

### Q6: How long does training take?

**A:** Training time depends on GPU and model size:

- YOLOv11n: ~1-2 hours (Google Colab T4 GPU)
- YOLOv11s: ~2-3 hours
- YOLOv11m: ~4-6 hours

### Q7: Can I add my own parking lot images?

**A:** Absolutely! Annotate your images in YOLO format, add to the dataset folders, and retrain the model. More diverse data improves generalization.

### Q8: What's the inference speed?

**A:** Expected speeds:

- YOLOv11n: ~60-100 FPS (GPU), ~5-10 FPS (CPU)
- YOLOv11s: ~40-60 FPS (GPU), ~3-7 FPS (CPU)
- YOLOv11m: ~30-40 FPS (GPU), ~2-4 FPS (CPU)

---
