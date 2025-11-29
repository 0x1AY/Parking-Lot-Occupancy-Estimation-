# Multi-Stage Parking Lot Detection Pipeline

## Overview

Robust pipeline for detecting parking lot occupancy at any scale by first localizing the parking lot, then performing targeted high-resolution detection.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: Lot Localization                     │
│  Input: Coordinates → Wide-area image → Parking lot segmentation│
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                Stage 2: Coverage Planning                        │
│     Parking lot boundary → Calculate tile grid → API requests   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 3: High-Res Tile Detection                    │
│    Download tiles → Run YOLOv11 on each → Collect detections    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           Stage 4: Stitching & Aggregation                       │
│  Merge tiles → NMS across boundaries → Final occupancy report   │
└─────────────────────────────────────────────────────────────────┘
```

## Stage 1: Parking Lot Localization

### Purpose

Identify exact parking lot boundaries from wide-area satellite imagery

### Approaches

#### Option A: Parking Lot Detection Model

- **Model**: Instance/semantic segmentation (Mask R-CNN, U-Net, SAM)
- **Training**: Use APKLOT dataset (500 images, 7000+ parking lot polygons)
- **Input**: Wide-area image (e.g., 2048x2048 at low zoom)
- **Output**: Polygon/mask defining parking lot boundary

#### Option B: Interactive Annotation (Current Manual Method)

- User draws bounding box on map interface
- Simpler but less automated

#### Option C: Hybrid Approach

- Model suggests parking lot boundary
- User reviews/adjusts if needed
- Best of both worlds

### Implementation Details

```python
def localize_parking_lot(lat, lon, api_key):
    """
    Stage 1: Detect parking lot boundaries
    """
    # Fetch wide-area image (low zoom for context)
    wide_image = fetch_static_map(
        lat=lat,
        lon=lon,
        zoom=18,  # Lower zoom = wider area
        size=(2048, 2048),
        scale=1
    )

    # Run parking lot detection model
    # (Trained on APKLOT dataset)
    lot_mask = parking_lot_detector.predict(wide_image)

    # Convert mask to polygon
    parking_lot_polygon = mask_to_polygon(lot_mask)

    return parking_lot_polygon, wide_image
```

## Stage 2: Coverage Planning

### Purpose

Calculate optimal tile grid to cover detected parking lot area

### Algorithm

```python
def plan_tile_coverage(parking_lot_polygon, target_resolution=0.3):
    """
    Stage 2: Calculate tile grid to cover parking lot

    Args:
        parking_lot_polygon: Detected lot boundary
        target_resolution: Meters per pixel (0.3m for high detail)

    Returns:
        List of tile requests (lat, lon, zoom, size)
    """
    # Get bounding box of parking lot
    min_lat, min_lon, max_lat, max_lon = parking_lot_polygon.bounds

    # Calculate tile size at target resolution
    zoom = calculate_zoom_for_resolution(target_resolution)
    tile_size_meters = 640 * target_resolution  # 192m x 192m per tile

    # Calculate number of tiles needed
    lat_span_m = haversine_distance(min_lat, min_lon, max_lat, min_lon)
    lon_span_m = haversine_distance(min_lat, min_lon, min_lat, max_lon)

    tiles_lat = math.ceil(lat_span_m / (tile_size_meters * 0.8))  # 20% overlap
    tiles_lon = math.ceil(lon_span_m / (tile_size_meters * 0.8))

    # Generate tile grid
    tiles = []
    for i in range(tiles_lat):
        for j in range(tiles_lon):
            tile_lat = min_lat + (i * tile_size_meters * 0.8) / 111320
            tile_lon = min_lon + (j * tile_size_meters * 0.8) / (111320 * math.cos(math.radians(tile_lat)))

            tiles.append({
                'lat': tile_lat,
                'lon': tile_lon,
                'zoom': zoom,
                'size': (640, 640),
                'scale': 2,
                'tile_id': f'tile_{i}_{j}'
            })

    return tiles
```

### Tile Overlap Strategy

- **20% overlap** between adjacent tiles
- Ensures no detections are lost at boundaries
- Allows for robust NMS across tiles

## Stage 3: High-Resolution Detection

### Purpose

Run YOLOv11 detection on each tile individually

### Implementation

```python
def detect_on_tiles(tiles, model, api_key):
    """
    Stage 3: Run detection on each tile
    """
    all_detections = []

    for tile in tiles:
        # Download tile
        img = fetch_static_map(
            lat=tile['lat'],
            lon=tile['lon'],
            zoom=tile['zoom'],
            size=tile['size'],
            scale=tile['scale'],
            api_key=api_key
        )

        # Run YOLOv11 detection
        results = model.predict(img, conf=0.25)

        # Convert to global coordinates
        for detection in results[0].boxes:
            det = {
                'class': int(detection.cls),
                'conf': float(detection.conf),
                'bbox': detection.xyxy[0].tolist(),  # Local coords
                'tile_id': tile['tile_id'],
                'tile_origin': (tile['lat'], tile['lon'])
            }

            # Convert bbox to global lat/lon
            det['bbox_global'] = local_to_global_coords(
                det['bbox'],
                tile['lat'],
                tile['lon'],
                tile['zoom'],
                tile['size']
            )

            all_detections.append(det)

    return all_detections
```

## Stage 4: Stitching & Aggregation

### Purpose

Merge detections from overlapping tiles and compute final occupancy

### Challenges

- **Duplicate detections** at tile boundaries
- **Coordinate system alignment**
- **Confidence score harmonization**

### Solution: Global NMS

```python
def stitch_and_aggregate(all_detections, iou_threshold=0.5):
    """
    Stage 4: Merge tile detections and calculate occupancy
    """
    # Group by class
    detections_by_class = {}
    for det in all_detections:
        cls = det['class']
        if cls not in detections_by_class:
            detections_by_class[cls] = []
        detections_by_class[cls].append(det)

    # Apply NMS within each class
    final_detections = {}
    for cls, dets in detections_by_class.items():
        # Convert to global coordinate bboxes
        bboxes = np.array([d['bbox_global'] for d in dets])
        scores = np.array([d['conf'] for d in dets])

        # Global NMS
        keep_indices = nms(bboxes, scores, iou_threshold)
        final_detections[cls] = [dets[i] for i in keep_indices]

    # Calculate occupancy
    cars = final_detections.get(0, [])  # class 0 = car
    stalls = final_detections.get(3, [])  # class 3 = stall

    occupancy = calculate_occupancy(cars, stalls)

    return {
        'detections': final_detections,
        'occupancy': occupancy,
        'total_stalls': len(stalls),
        'occupied_stalls': occupancy['occupied'],
        'occupancy_rate': occupancy['rate']
    }
```

### Visualization

```python
def visualize_stitched_results(parking_lot_polygon, final_detections, wide_image):
    """
    Create comprehensive visualization
    """
    fig, ax = plt.subplots(1, figsize=(20, 20))

    # Show wide-area image
    ax.imshow(wide_image)

    # Draw parking lot boundary
    boundary_coords = parking_lot_polygon.exterior.coords
    boundary_xy = [global_to_image_coords(lat, lon) for lat, lon in boundary_coords]
    ax.plot(*zip(*boundary_xy), 'yellow', linewidth=3, label='Parking Lot Boundary')

    # Draw detections
    for cls, dets in final_detections.items():
        color = COLORS[cls]
        for det in dets:
            bbox = det['bbox_global']
            rect = patches.Rectangle(...)
            ax.add_patch(rect)

    plt.legend()
    plt.title(f"Parking Lot Occupancy: {occupancy_rate:.1f}%")
    plt.savefig('full_parking_lot_analysis.png', dpi=150)
```

## Research References

### Key Papers Implementing Similar Approaches

1. **APKLOT Dataset Paper** (WACV 2022)

   - Y. Yin et al., "A Context-Enriched Satellite Imagery Dataset and an Approach for Parking Lot Detection"
   - Uses two-stage approach: lot localization → vehicle detection
   - APKLOT dataset: 500 satellite images with parking lot polygons

2. **Parking Space Inventory from Above** (IET ITS 2023)

   - J. Hellekes et al.
   - Detection on aerial images + estimation for unobserved regions
   - Multi-scale analysis approach

3. **Parking Occupancy on PlanetScope Satellite** (Remote Sensing 2023)

   - S. Drouyer
   - Planet Labs satellite imagery (3-5m resolution)
   - Large-area coverage strategy

4. **Stereo Satellite Images** (Remote Sensing 2020)
   - S. Zambanini et al.
   - Parking detection using stereo satellite pairs
   - Height information for better segmentation

## Advantages Over Single-Image Approach

| Aspect                       | Single Centered Image       | Multi-Stage Tiled Approach   |
| ---------------------------- | --------------------------- | ---------------------------- |
| **Coverage**                 | Fixed (640x640)             | Adaptive to lot size         |
| **Resolution**               | Limited by API              | Higher (more pixels per lot) |
| **Large Lots**               | May cut off edges           | Complete coverage            |
| **Small Lots**               | Wastes area on surroundings | Focused only on lot          |
| **Irregular Shapes**         | Poor fit                    | Perfect fit                  |
| **Computational Efficiency** | Processes unnecessary areas | Only parking lot regions     |
| **Scalability**              | Fixed                       | Scales with lot size         |

## Implementation Roadmap

### Phase 1: Parking Lot Detection Model (NEW)

- [ ] Train segmentation model on APKLOT dataset
- [ ] Evaluate on test set
- [ ] Deploy for inference

### Phase 2: Tile Planning System

- [ ] Implement coverage calculation
- [ ] Add overlap strategy
- [ ] Optimize tile count

### Phase 3: Detection Pipeline

- [ ] Parallel tile downloads
- [ ] Batch detection on tiles
- [ ] Coordinate transformations

### Phase 4: Stitching & NMS

- [ ] Global NMS implementation
- [ ] Occupancy aggregation
- [ ] Visualization

### Phase 5: Testing & Validation

- [ ] Test on Walmart locations
- [ ] Compare with single-image approach
- [ ] Performance benchmarking

## Next Steps

1. **Review APKLOT paper** - Get exact methodology details
2. **Download APKLOT dataset** - Train parking lot detection model
3. **Implement Stage 1** - Lot localization
4. **Test on Walmart** - Validate approach
5. **Full pipeline** - Integrate all stages

## Notes

- Your multi-class model already detects `lot_boundary` (class 1) - could be used!
- APKLOT has 7,000+ parking lot polygons - perfect for training
- This approach is mentioned in your proposal references
- Scalable to any parking lot size worldwide
