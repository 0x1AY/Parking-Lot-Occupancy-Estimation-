# APKLOT Dataset Exploration

**Date**: November 28, 2025  
**Repository**: https://github.com/langheran/APKLOT  
**Clone Location**: `/Users/ay/Desktop/deeplearning/Parking lot /APKLOT`

---

## Dataset Overview

### Repository Statistics

- **Total Size**: 2.75 GB
- **Files**: 11,432 files
- **Images**: 491 satellite parking lot images
- **Format**: LabelMe JSON + Pascal VOC XML + Segmentation Masks

### Key Directories

```
APKLOT/
├── 1. Satellite/              # Satellite imagery dataset (MAIN)
│   ├── Dataset/
│   │   ├── labelme_20/        # 500+ LabelMe JSON annotations (with embedded images)
│   │   ├── World/             # Processed dataset
│   │   │   ├── training/      # 800 training files
│   │   │   ├── testing/       # 200 test files
│   │   │   └── PASCAL_format/ # Pascal VOC format
│   │   │       ├── JPEGImages/          # 491 satellite images
│   │   │       ├── Annotations/         # 491 XML annotations
│   │   │       ├── SegmentationClass/   # Segmentation masks
│   │   │       └── ImageSets/Segmentation/
│   │   │           ├── train.txt        # 299 training images
│   │   │           └── val.txt          # 100 validation images
│   │   ├── ITESM_20/          # Instituto Tecnológico campus dataset
│   │   ├── ITESM_21/          # Campus dataset variant
│   │   ├── WorldX0.4/         # Scaled variants
│   │   └── WorldX0.667/       # Scaled variants
│   ├── Scripts/
│   │   ├── 0. download/       # Google Maps downloading scripts
│   │   ├── 1. build_training_test_folders/  # Dataset splitting
│   │   ├── 2. pascal/         # Pascal VOC conversion
│   │   ├── 3. jittering/      # Data augmentation
│   │   └── 4. evaluation/     # IoU metrics
│   └── Description/           # Documentation and examples
│
└── 2. Camera/                 # Ground-level camera perspective
    ├── segmentation_1/
    └── segmentation_2/
```

---

## Dataset Details

### Image Characteristics

**From Pascal VOC XML (sample: 101881731.xml)**:

```xml
<size>
  <width>1482</width>
  <height>1373</height>
  <depth>3</depth>
</size>
```

- **Typical size**: ~1400x1400 pixels
- **Format**: JPEG for images, PNG for masks
- **File size**: 43 KB - 376 KB per image
- **Color**: RGB (3 channels)

### Annotation Format

**Class**: `parkingspot` (parking block/lot boundary)

**Pascal VOC bounding boxes**:

```xml
<object>
  <name>parkingspot</name>
  <pose>Frontal</pose>
  <truncated>0</truncated>
  <difficult>0</difficult>
  <bndbox>
    <xmin>632</xmin>
    <ymin>184</ymin>
    <xmax>1001</xmax>
    <ymax>270</ymax>
  </bndbox>
</object>
```

**LabelMe JSON polygons** (in labelme_20/):

- Contains polygon vertices for precise boundaries
- Has embedded `imageData` (base64 encoded)
- Label: "1" (parking block class)
- Average 15-35 polygons per image

### Train/Val/Test Split

**World dataset split**:

- **Training**: 299 images
- **Validation**: 100 images
- **Test**: ~92 images (491 total - 399 train/val)

**Geographic coverage**:

- Mexico (México City, Monterrey, Guadalajara)
- USA (New York, Los Angeles, Chicago, Houston)
- Chile (Santiago)
- Spain (Madrid)
- Japan (Tokyo)

---

## Data Format Analysis

### 1. LabelMe JSON Format

**Location**: `1. Satellite/Dataset/labelme_20/`

**Structure**:

```json
{
  "shapes": [
    {
      "label": "1",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon"
    }
  ],
  "imagePath": "101881731.json",
  "imageData": "<base64_encoded_image>",
  "imageHeight": 1373,
  "imageWidth": 1482
}
```

**Pros**:

- ✅ Contains polygon boundaries (not just bboxes)
- ✅ Embedded images (no separate file needed)
- ✅ Easy to parse with Python

**Cons**:

- ❌ Not YOLO format
- ❌ Need conversion for YOLOv11

### 2. Pascal VOC Format

**Location**: `1. Satellite/Dataset/World/PASCAL_format/`

**Components**:

- `JPEGImages/` - Original satellite images (491 files)
- `Annotations/` - XML files with bounding boxes (491 files)
- `SegmentationClass/` - PNG masks with class labels
- `SegmentationObject/` - PNG masks with instance labels
- `ImageSets/Segmentation/` - train.txt, val.txt splits

**Pros**:

- ✅ Standard format (widely supported)
- ✅ Segmentation masks ready
- ✅ Pre-split train/val sets

**Cons**:

- ❌ Bounding boxes only (not precise polygons)
- ❌ Still need YOLO conversion

---

## Conversion Requirements

### Target: YOLO Segmentation Format

For YOLOv11 segmentation training, we need:

```
datasets/
└── apklot/
    ├── images/
    │   ├── train/
    │   │   ├── 101881731.jpg
    │   │   └── ...
    │   └── val/
    │       ├── 103109948.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── 101881731.txt
        │   └── ...
        └── val/
            ├── 103109948.txt
            └── ...
```

**YOLO Segmentation Format** (labels/\*.txt):

```
<class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ... <xn_norm> <yn_norm>
```

Where:

- `class_id`: 0 (parking_lot)
- `x_norm, y_norm`: Normalized coordinates (0.0 to 1.0)
- Multiple lines for multiple parking lots in same image

---

## Conversion Strategy

### Option 1: LabelMe JSON → YOLO (RECOMMENDED)

**Source**: `labelme_20/*.json`  
**Advantage**: Has precise polygon coordinates

```python
import json
import base64
from PIL import Image
import io

def labelme_to_yolo(json_path, output_img_path, output_label_path):
    with open(json_path) as f:
        data = json.load(f)

    # Extract image from base64
    img_data = base64.b64decode(data['imageData'])
    img = Image.open(io.BytesIO(img_data))
    img.save(output_img_path)

    # Get image dimensions
    img_w, img_h = img.size

    # Convert polygons to YOLO format
    with open(output_label_path, 'w') as out:
        for shape in data['shapes']:
            if shape['label'] == '1':  # Parking lot class
                # Normalize coordinates
                points = []
                for x, y in shape['points']:
                    points.append(f"{x/img_w:.6f} {y/img_h:.6f}")

                # Write: class_id + normalized polygon coords
                out.write(f"0 {' '.join(points)}\n")
```

### Option 2: Pascal VOC Masks → YOLO

**Source**: `PASCAL_format/SegmentationClass/*.png`  
**Advantage**: Images already extracted

```python
import cv2
import numpy as np

def mask_to_yolo_polygon(mask_path, img_path, output_label_path):
    # Read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)
    h, w = mask.shape

    # Find contours (parking lot boundaries)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with open(output_label_path, 'w') as out:
        for contour in contours:
            # Simplify polygon
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Normalize coordinates
            points = []
            for point in approx:
                x, y = point[0]
                points.append(f"{x/w:.6f} {y/h:.6f}")

            # Write to file
            out.write(f"0 {' '.join(points)}\n")
```

---

## Next Steps

### Immediate Tasks

1. **Create conversion script**: `tools/convert_apklot_to_yolo.py`
   - Convert LabelMe JSON to YOLO segmentation format
   - Extract images from base64 or copy from Pascal format
   - Preserve train/val split from `ImageSets/Segmentation/`
2. **Create dataset YAML**: `data/apklot.yaml`

   ```yaml
   path: /Users/ay/Desktop/deeplearning/Parking lot /datasets/apklot
   train: images/train
   val: images/val

   nc: 1
   names: ["parking_lot"]
   ```

3. **Test conversion on sample images**:
   - Convert 5 training images
   - Visualize to verify polygons are correct
   - Check for any formatting issues

### Training Setup

4. **Configure YOLOv11 segmentation training**:

   ```bash
   yolo segment train \
     data=data/apklot.yaml \
     model=yolov11m-seg.pt \
     epochs=100 \
     imgsz=640 \
     batch=16 \
     project=parking_lot_localization \
     name=apklot_stage1
   ```

5. **Training parameters**:
   - Use YOLOv11m-seg (medium model for balance)
   - Image size: 640 (APKLOT images are ~1400px, will be resized)
   - Augmentation: Enable mosaic, flip, rotate
   - Epochs: 100+ (monitor validation mAP)

### Validation

6. **Test trained model on wide-area images**:

   - Download Walmart image at zoom 18 (2048x2048)
   - Run parking lot detection
   - Verify complete lots are detected

7. **Compare with your existing model**:
   - Your model: Trained on zoom 20 tiles (640x640)
   - New model: Trained on APKLOT wide-area images
   - Expected: New model detects complete parking lots

---

## Key Insights

### Why APKLOT Solves Your Problem

1. **Scale mismatch resolved**:

   - Your current model: Single tiles @ zoom 20 (narrow view)
   - APKLOT images: Wide-area @ ~1400px (complete lots visible)
   - Training on APKLOT = model sees full parking lot context

2. **Perfect for multi-stage pipeline**:

   - **Stage 1**: APKLOT-trained model detects parking lot on wide image (zoom 18)
   - **Stage 2**: Your existing model detects cars/stalls on high-res tiles (zoom 20)

3. **Global diversity**:
   - 10+ cities across 5 countries
   - Validates generalization to unseen locations
   - Similar to your Walmart test (Toronto locations)

### Dataset Advantages

✅ **500 images** with **7,000+ parking lot annotations**  
✅ **MIT License** - fully open source  
✅ **Multiple formats** - LabelMe JSON (polygons) + Pascal VOC (masks)  
✅ **Pre-split** train/val/test sets  
✅ **Global coverage** - diverse parking lot types  
✅ **Google Maps source** - matches your data pipeline  
✅ **Augmentation scripts** included - can expand dataset

---

## Comparison: APKLOT vs Your Current Data

| Feature            | Your Dataset                          | APKLOT                           |
| ------------------ | ------------------------------------- | -------------------------------- |
| **Images**         | ~1,000 tiles                          | 491 wide-area images             |
| **Zoom level**     | 20 (high-res)                         | ~18-19 (wide-area)               |
| **Image size**     | 640x640                               | ~1400x1400                       |
| **Classes**        | 4 (car, lot_boundary, objects, stall) | 1 (parking_lot)                  |
| **Annotation**     | Bounding boxes                        | Polygons + Masks                 |
| **Coverage**       | BC, Canada                            | Global (10+ cities)              |
| **Use case**       | Vehicle/stall detection               | Parking lot localization         |
| **Pipeline stage** | Stage 2 (high-res detection)          | Stage 1 (wide-area localization) |

**Complementary**: APKLOT provides Stage 1 capability, your model provides Stage 2!

---

## Resources

### Documentation

- **README**: `/Users/ay/Desktop/deeplearning/Parking lot /APKLOT/README.md`
- **GitHub**: https://github.com/langheran/APKLOT
- **Paper**: "Robust Parking Block Segmentation from a Surveillance Camera Perspective" (Applied Sciences 2020)

### Scripts (Jupyter Notebooks)

- **Download**: `Scripts/0. download/` - Google Maps API downloading
- **Format conversion**: `Scripts/2. pascal/` - 5 notebooks for Pascal VOC
- **Augmentation**: `Scripts/3. jittering/` - Data augmentation with imgaug
- **Evaluation**: `Scripts/4. evaluation/` - IoU metrics calculation

### Data Paths

- **LabelMe JSON**: `1. Satellite/Dataset/labelme_20/` (500+ files)
- **Pascal VOC Images**: `1. Satellite/Dataset/World/PASCAL_format/JPEGImages/` (491 images)
- **Annotations**: `1. Satellite/Dataset/World/PASCAL_format/Annotations/` (491 XML)
- **Masks**: `1. Satellite/Dataset/World/PASCAL_format/SegmentationClass/` (491 PNG)
- **Train split**: `1. Satellite/Dataset/World/PASCAL_format/ImageSets/Segmentation/train.txt`
- **Val split**: `1. Satellite/Dataset/World/PASCAL_format/ImageSets/Segmentation/val.txt`

---

## Action Plan

### Phase 1: Conversion (Today)

- [ ] Write `tools/convert_apklot_to_yolo.py`
- [ ] Create YOLO directory structure
- [ ] Convert train split (299 images)
- [ ] Convert val split (100 images)
- [ ] Create `data/apklot.yaml`

### Phase 2: Training (This Week)

- [ ] Train YOLOv11m-seg on APKLOT
- [ ] Monitor training metrics (mAP, loss)
- [ ] Validate on APKLOT test set
- [ ] Export best model weights

### Phase 3: Integration (Next Week)

- [ ] Test on Walmart wide-area images (zoom 18)
- [ ] Implement tile planning from detected polygons
- [ ] Build multi-stage pipeline
- [ ] Compare with single-stage approach

### Phase 4: Evaluation (Following Week)

- [ ] Benchmark accuracy improvement
- [ ] Measure coverage (missed lots)
- [ ] Calculate API cost efficiency
- [ ] Document results

---

**Status**: ✅ Dataset cloned and explored  
**Next**: Create YOLO conversion script  
**Goal**: Train Stage 1 parking lot localization model for multi-stage pipeline
