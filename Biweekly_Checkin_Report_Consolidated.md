# Biweekly Check-in Report

**Group:** Parking Lot Occupancy Estimation  
**Date:** November 28 - December 5, 2025  
**Team Members:** Aminu Yiwere, Olatunji Olagundoye

---

## 1. What Have You Done?

### Complete Parking Occupancy Detection System with Dual-Model Architecture

We successfully developed and deployed a production-ready parking lot occupancy estimation system that addresses the critical limitation of single-tile detection models and implements precise dual-model occupancy calculation.

#### Key Accomplishments:

**A. APKLOT Dataset Integration & Parking Lot Localization Model**

Training completed in 47 minutes on Google Colab T4 GPU:
- Cloned and processed the APKLOT dataset (500 satellite images, 7,000+ parking lot polygons)
- Converted 391 images from LabelMe JSON format to YOLO segmentation format
- Trained YOLOv11m-seg model for wide-area parking lot localization
- Achieved strong performance metrics:
  - Box mAP50: 83.5%, mAP50-95: 62.4%
  - Mask mAP50: 76.1%, mAP50-95: 43.7%
  - Inference speed: 14.8ms per image

**B. Optimal Zoom Level Testing & Validation**

Tested model performance across three zoom levels:
- Zoom 20 (160m coverage): 86 parking lots detected, 8.6 avg per image
- Zoom 19 (320m coverage): 191 parking lots detected, 19.1 avg per image ✓ **OPTIMAL**
- Zoom 18 (640m coverage): 416 parking lots detected, 41.6 avg per image (too wide)
- Determined Zoom 19 as optimal balance between coverage and detection precision
- Achieved 100% success rate across all 10 Walmart test locations

**C. Dual-Model Architecture Implementation (96.3% + 84% mAP50)**

Developed and trained two specialized detection models:

**Model 1 - Car Detection (YOLOv11m):**
- Achieved **96.3% mAP50** on vehicle detection
- Dataset: 1,109 annotated satellite images
- Single-class detection (vehicles only)
- Inference: 14.7ms per image
- Purpose: High-accuracy vehicle localization

**Model 2 - Stall Detection (YOLOv11m):**
- Achieved **84% mAP50** on parking stall detection  
- Multi-class: occupied stalls, vacant stalls, handicap spaces
- Dataset: Same satellite imagery with stall annotations
- Purpose: Precise parking space boundaries

**IoU-Based Occupancy Algorithm:**
- Matches detected cars to parking stalls using Intersection over Union (IoU)
- Threshold: 0.3 for optimal car-to-stall matching
- Handles various parking angles and vehicle sizes
- Clear separation: occupied (red) vs. vacant (green) visualization

**D. Complete Four-Stage Pipeline Implementation**

Developed end-to-end pipeline addressing single-tile limitations:

**Stage 1 - Parking Lot Localization:**
- Detects parking lot areas from wide-area satellite imagery (zoom 19)
- Extracts bounding box coordinates for each detected parking lot
- Calculates geographic bounds and area dimensions
- Solves the tile-splitting problem by identifying full parking lot extent

**Stage 2 - Tile Coverage Planning:**
- Converts pixel coordinates to latitude/longitude
- Plans optimal tile grid with 20% overlap for seamless stitching
- Downloads high-resolution tiles (zoom 20, 640x640@2x) for detected areas only
- Efficient: Only downloads tiles covering parking lots (not entire region)

**Stage 3 - Dual Model Detection:**
- Runs both YOLOv11m models (car + stall) on each tile
- Processes tiles with GPU acceleration
- Aggregates detection results across all tiles
- IoU matching determines which stalls are occupied

**Stage 4 - Result Stitching & Visualization:**
- Stitches tiles into complete parking lot view
- Overlays all detections with color coding (green/red)
- Generates comprehensive occupancy statistics
- Produces both visualizations and JSON reports

**E. Comprehensive Batch Processing & Validation**

Successfully processed 10 major retail locations across Greater Toronto Area:
- Walmart Gerrard St, Dufferin St, St Clair Ave, Islington Ave, Lawrence Ave
- Walmart Pickering, Brampton, Scarborough, Ajax, Markham

**Aggregate Results:**
- **813 parking stalls detected** across all 10 locations
- **226 occupied stalls** (27.8% average occupancy)
- **587 vacant stalls** (72.2% availability)
- **100% pipeline success rate** (no processing failures)
- Processing time: 30-60 seconds per location

**F. Tools & Scripts Created**

1. `tools/convert_apklot_to_yolo.py` - Dataset conversion utility
2. `tools/visualize_apklot.py` - Annotation visualization
3. `tools/test_parking_lot_detection.py` - Model testing and evaluation
4. `tools/download_zoom18_test.py` & `download_zoom19_test.py` - Multi-zoom testing
5. `tools/plan_tile_coverage.py` - Geographic tile planning algorithm
6. `occupancy/unified_parking_pipeline.py` - Complete end-to-end pipeline
7. `occupancy/batch_process_walmart_locations.py` - Automated batch processing

**G. Additional Feature - Web Application (Extra Credit)**

As an extra deliverable beyond project requirements, developed a user-friendly web interface:
- Built with Streamlit for accessibility to non-technical users
- Simple lat/lon input for any parking lot worldwide
- Real-time progress tracking through 4 processing stages
- Interactive visualization with downloadable results
- Deployed on Streamlit Cloud for public access
- Includes comprehensive documentation and user guides

*Note: The web application is a supplementary feature demonstrating practical deployment, not a core project requirement.*

---

## 2. What Did You Learn?

### Key Technical Insights

**A. Dataset Quality Critical for Segmentation:**
- APKLOT's diverse parking lot geometries improved model generalization
- Polygon annotations enable precise boundary detection vs. bounding boxes
- Mix of commercial, residential, and institutional parking lots crucial for robustness

**B. Zoom Level Significantly Impacts Detection:**
- Tested Zoom 18, 19, and 20 systematically
- Zoom 19 provides optimal balance: coverage (320m) + detail for detection
- Too wide (Zoom 18) causes false positives; too narrow (Zoom 20) misses lots
- Context window size directly affects model's ability to distinguish parking lots

**C. Tile-Based Processing Solves Scale Limitations:**
- Single-tile approaches fail when parking lots exceed tile boundaries
- Stage 1 localization at lower zoom identifies full parking lot extent
- Stage 2 tile planning with 20% overlap ensures seamless stitching
- Overlap critical for handling detections at tile edges

**D. Dual-Model Architecture Superior to Single-Model:**
- Initial single-model approach: 75% mAP50 (cars + stalls combined)
- Separated models: 96.3% (cars) + 84% (stalls) significantly more accurate
- Specialized models learn distinct features (moving objects vs. static markings)
- IoU matching provides precise occupancy calculation vs. classification

**E. Geographic Coordinate Transformation Non-Trivial:**
- Google Maps Static API uses Mercator projection (EPSG:3857)
- Tile boundaries require precise lat/lon to pixel conversion
- 20% overlap handled by expanding bounding boxes before tile planning
- Scale factor (2x) must be accounted for in all coordinate calculations

---

## 3. What Challenges Did You Encounter?

### A. Initial Dataset Challenges

**Challenge:** APKLOT dataset in LabelMe JSON format incompatible with YOLO  
**Solution:** Created conversion script handling:
- Polygon coordinate extraction and normalization
- Image dimension mapping
- Label format conversion (category_id → class index)
- Validated all 391 converted annotations

**Challenge:** Class imbalance in APKLOT (parking lots vs. background)  
**Solution:** Applied augmentation (flip, rotation, brightness) during training

### B. Tile Coverage and Stitching Issues

**Challenge:** Parking lots split across tile boundaries caused incomplete detections  
**Solution:** Implemented two-stage localization:
1. Wide-area detection (Zoom 19) identifies full parking lot
2. High-resolution tiles (Zoom 20) with 20% overlap for detailed detection

**Challenge:** Coordinate system mismatch between API and image pixels  
**Solution:** Developed precise transformation functions:
- `latlon_to_pixel()` for geographic to image space
- `pixel_to_latlon()` for inverse transformation
- Accounting for scale factor (640x640@2x = 1280x1280 actual pixels)

### C. Dual-Model Integration

**Challenge:** Matching detections from two separate models  
**Solution:** IoU-based matching algorithm:
- Calculate Intersection over Union for car-stall pairs
- Threshold: 0.3 balances precision and recall
- Handles partial overlaps and angled parking

**Challenge:** Duplicate detections across overlapping tiles  
**Solution:** Implemented Non-Maximum Suppression (NMS):
- IoU threshold: 0.45 for duplicate removal
- Keeps highest confidence detection per unique object

### D. Web Application Deployment (Extra Feature)

**Challenge:** OpenCV ImportError on Streamlit Cloud (libGL.so.1 missing)  
**Solution:** 
- Created `packages.txt` with system dependencies (libgl1-mesa-glx, libglib2.0-0)
- Switched to `opencv-python-headless` (no GUI dependencies)

**Challenge:** API key security and configuration  
**Solution:** Implemented triple-layer API key loading:
1. Environment variables (.env file for local development)
2. Streamlit secrets (cloud deployment)
3. Manual input field (fallback option)

**Challenge:** Google Maps API 403 Forbidden errors  
**Solution:** Added comprehensive troubleshooting to error messages:
- Check Maps Static API enabled in Google Cloud Console
- Verify API key restrictions
- Confirm billing account active

---

## 4. What Are Your Next Steps?

### Completed Milestones (from Nov 28 Planning)

✅ **Enhanced Occupancy Analysis** - Implemented dual-model architecture (96.3% + 84% mAP50)  
✅ **Duplicate Detection Removal** - NMS with IoU threshold (0.45)  
✅ **Performance Optimization** - Achieved 30-60 second processing per location  
✅ **Batch Processing Capability** - Successfully processed 10 locations (813 stalls)

### Future Enhancements (Beyond Current Scope)

**A. Temporal Analysis:**
- Track occupancy patterns over time (hourly, daily, seasonal)
- Identify peak usage times and trends
- Predictive modeling for future occupancy

**B. Multi-Location Dashboard:**
- Web interface showing multiple parking lots simultaneously
- Comparative occupancy statistics
- Heat maps showing utilization patterns

**C. Real-Time Monitoring:**
- Integrate with live satellite feeds for near real-time updates
- Alert system for capacity thresholds
- Historical trend visualization

**D. Model Performance Enhancement:**
- Fine-tune stall detection model for better edge case handling
- Experiment with YOLOv11x (larger model) for potential accuracy gains
- Test on diverse geographic regions beyond GTA

---

## 5. Summary Statistics

**Development Effort:**
- Total development time: ~8 hours (data prep, training, pipeline implementation)
- APKLOT training time: 47 minutes (T4 GPU)
- Car detection training: ~2 hours
- Stall detection training: ~2 hours
- Total code: ~1,500 lines (conversion scripts, pipeline, batch processing)

**Model Performance:**
- Localization model: 83.5% mAP50 (box), 76.1% mAP50 (mask)
- Car detection model: 96.3% mAP50
- Stall detection model: 84% mAP50
- Combined system: 100% success rate on 10 test locations

**Production Results:**
- 813 parking stalls analyzed across 10 locations
- 226 occupied (27.8%), 587 vacant (72.2%)
- Average processing time: 30-60 seconds per location
- Zero pipeline failures in batch processing

**Technical Resources:**
- 3 trained models (121MB total):
  - Localization: 43MB (`datasets/apklot/apklot_stage1/weights/best.pt`)
  - Car detection: 39MB (`parking_runs/yolo11m_parking_augmented2/weights/best.pt`)
  - Stall detection: 39MB (`parking_runs/yolo11m_multilabel/weights/best.pt`)
- 7 utility scripts for data processing and testing
- Complete documentation (README, deployment guides, API docs)

---

## 6. Technical Implementation Details

### Core Pipeline Architecture

```
Input: Latitude, Longitude
  ↓
Stage 1: Parking Lot Localization (Zoom 19, YOLOv11m-seg)
  → Detects parking lot boundaries
  → Extracts bounding boxes
  ↓
Stage 2: Tile Coverage Planning
  → Converts bbox to lat/lon bounds
  → Plans tile grid (Zoom 20, 640x640@2x, 20% overlap)
  → Downloads tiles via Google Maps Static API
  ↓
Stage 3: Dual Model Detection
  → Car Detection (YOLOv11m, 96.3% mAP50)
  → Stall Detection (YOLOv11m, 84% mAP50)
  → IoU Matching (threshold: 0.3)
  ↓
Stage 4: Result Stitching & Visualization
  → Stitch tiles into complete view
  → Color-coded overlay (green/red)
  → Generate statistics + JSON report
  ↓
Output: Occupancy metrics, visualization, detection data
```

### Key Algorithms

**1. IoU-Based Occupancy Matching:**
```
For each car detection:
  For each stall detection:
    Calculate IoU = Intersection Area / Union Area
    If IoU > 0.3:
      Mark stall as occupied
      Break (car matched to stall)
```

**2. Tile Planning with Overlap:**
```
bbox = parking_lot_bounds
expanded_bbox = expand(bbox, overlap=0.2)
tiles = []
for lat in range(north, south, tile_height * 0.8):
  for lon in range(west, east, tile_width * 0.8):
    tiles.append(download_tile(lat, lon, zoom=20))
```

**3. Non-Maximum Suppression (NMS):**
```
For each detection pair:
  If IoU > 0.45 and same_class:
    Keep detection with higher confidence
    Discard lower confidence duplicate
```

---

## Conclusion

This project successfully delivers a complete, production-ready parking lot occupancy estimation system that overcomes the limitations of single-tile detection approaches. The dual-model architecture (96.3% + 84% mAP50) combined with intelligent tile planning and IoU-based occupancy calculation provides accurate, scalable analysis of parking lots of any size.

The system has been validated on 813 real parking stalls across 10 major retail locations with a 100% success rate. The web application (extra feature) demonstrates practical deployment potential for end-users.

**Core Achievement:** Dual-model satellite-based parking occupancy detection system with 96.3% car detection and 84% stall detection accuracy, validated on 813 parking stalls.

**Bonus Feature:** Production web application deployed on Streamlit Cloud for user-friendly access.
