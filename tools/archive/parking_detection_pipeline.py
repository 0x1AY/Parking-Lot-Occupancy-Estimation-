#!/usr/bin/env python3
"""
Multi-Stage Parking Lot Detection Pipeline
===========================================
Complete end-to-end pipeline for parking occupancy estimation:

Stage 1: Wide-area parking lot localization (zoom 19)
Stage 2: High-resolution tile download for detected areas (zoom 20)
Stage 3: Vehicle/stall detection on tiles
Stage 4: Result stitching and occupancy calculation

Usage:
    python tools/parking_detection_pipeline.py --image path/to/wide_area.png --lat 43.668734 --lon -79.340158
    python tools/parking_detection_pipeline.py --location walmart_01
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json
import requests
from PIL import Image
from io import BytesIO
import time
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math


class ParkingDetectionPipeline:
    """End-to-end parking lot occupancy detection pipeline."""
    
    def __init__(self, 
                 localization_model_path: str = "datasets/apklot/apklot_stage1/weights/best.pt",
                 detection_model_path: str = "parking_runs/yolo11m_multiclass/weights/best.pt",
                 google_api_key: str = "AIzaSyCZWUlRCSb7WxHNBWtMifWRW25GOWfbous"):
        """
        Initialize pipeline with models.
        
        Args:
            localization_model_path: Path to APKLOT parking lot localization model
            detection_model_path: Path to vehicle/stall detection model
            google_api_key: Google Maps API key for tile downloads
        """
        self.api_key = google_api_key
        
        # Load Stage 1 model (parking lot localization)
        print(f"Loading localization model: {localization_model_path}")
        self.localization_model = YOLO(localization_model_path)
        print("✓ Localization model loaded")
        
        # Load Stage 2 model (vehicle detection)
        print(f"Loading detection model: {detection_model_path}")
        self.detection_model = YOLO(detection_model_path)
        print("✓ Detection model loaded")
    
    def calculate_meters_per_pixel(self, zoom: int, latitude: float) -> float:
        """Calculate meters per pixel at given zoom and latitude."""
        earth_circumference = 40075016.686
        lat_rad = math.radians(latitude)
        mpp_equator = earth_circumference / (2 ** (zoom + 8))
        return mpp_equator * math.cos(lat_rad)
    
    def pixel_to_latlon(self, x: float, y: float, 
                       img_width: int, img_height: int,
                       center_lat: float, center_lon: float,
                       zoom: int) -> Tuple[float, float]:
        """Convert pixel coordinates to lat/lon."""
        meters_per_pixel = self.calculate_meters_per_pixel(zoom, center_lat)
        
        dx_pixels = x - (img_width / 2)
        dy_pixels = (img_height / 2) - y
        
        dx_meters = dx_pixels * meters_per_pixel
        dy_meters = dy_pixels * meters_per_pixel
        
        lat_offset = dy_meters / 111319.9
        lon_offset = dx_meters / (111319.9 * math.cos(math.radians(center_lat)))
        
        return center_lat + lat_offset, center_lon + lon_offset
    
    def stage1_detect_parking_areas(self, image_path: Path, 
                                     center_lat: float, center_lon: float,
                                     zoom: int = 19, conf_threshold: float = 0.6) -> List[Dict]:
        """
        Stage 1: Detect parking lot areas from wide-area image.
        Returns bounding boxes (rectangles) for each parking lot.
        """
        print("\n" + "="*70)
        print("STAGE 1: Parking Lot Localization")
        print("="*70)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img_height, img_width = image.shape[:2]
        print(f"Image: {image_path.name}")
        print(f"Size: {img_width}x{img_height} pixels")
        print(f"Center: {center_lat:.6f}, {center_lon:.6f}")
        print(f"Zoom: {zoom}")
        
        # Run detection
        print(f"\nDetecting parking lots (conf >= {conf_threshold})...")
        results = self.localization_model.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=0.45,
            verbose=False,
            device='mps'
        )[0]
        
        num_detections = len(results.boxes) if results.boxes is not None else 0
        print(f"✓ Found {num_detections} parking lot(s)")
        
        if num_detections == 0:
            return []
        
        # Extract bounding boxes and convert to geographic coordinates
        parking_areas = []
        
        for idx, box in enumerate(results.boxes):
            conf = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert corners to lat/lon
            lat_top_left, lon_top_left = self.pixel_to_latlon(
                x1, y1, img_width, img_height, center_lat, center_lon, zoom
            )
            lat_bottom_right, lon_bottom_right = self.pixel_to_latlon(
                x2, y2, img_width, img_height, center_lat, center_lon, zoom
            )
            
            # Calculate area dimensions
            width_m = (lon_bottom_right - lon_top_left) * 111319.9 * math.cos(math.radians(center_lat))
            height_m = (lat_top_left - lat_bottom_right) * 111319.9
            
            area = {
                'id': idx + 1,
                'confidence': conf,
                'bbox_pixels': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                'bbox_geo': {
                    'min_lat': lat_bottom_right,
                    'max_lat': lat_top_left,
                    'min_lon': lon_top_left,
                    'max_lon': lon_bottom_right,
                    'center_lat': (lat_top_left + lat_bottom_right) / 2,
                    'center_lon': (lon_top_left + lon_bottom_right) / 2
                },
                'size_meters': {'width': abs(width_m), 'height': abs(height_m)}
            }
            
            print(f"\nParking Area #{idx + 1}:")
            print(f"  Confidence: {conf:.3f}")
            print(f"  Size: {abs(width_m):.1f}m × {abs(height_m):.1f}m")
            print(f"  Bounds: ({lat_bottom_right:.6f}, {lon_top_left:.6f}) to ({lat_top_left:.6f}, {lon_bottom_right:.6f})")
            
            parking_areas.append(area)
        
        return parking_areas
    
    def stage2_plan_and_download_tiles(self, parking_area: Dict, 
                                       output_dir: Path,
                                       tile_zoom: int = 20,
                                       tile_size: int = 640,
                                       overlap: float = 0.2) -> List[Dict]:
        """
        Stage 2: Plan tile grid and download high-resolution tiles.
        Returns list of downloaded tile paths with their coordinates.
        """
        print("\n" + "="*70)
        print(f"STAGE 2: Tile Download (Parking Area #{parking_area['id']})")
        print("="*70)
        
        bbox = parking_area['bbox_geo']
        area_id = parking_area['id']
        
        # Calculate tile coverage
        center_lat = bbox['center_lat']
        meters_per_pixel = self.calculate_meters_per_pixel(tile_zoom, center_lat)
        
        # Calculate tile coverage in degrees
        tile_meters = tile_size * 2 * meters_per_pixel  # scale=2 for retina
        tile_lat_deg = tile_meters / 111319.9
        tile_lon_deg = tile_meters / (111319.9 * math.cos(math.radians(center_lat)))
        
        # Calculate step size with overlap
        step_lat = tile_lat_deg * (1 - overlap)
        step_lon = tile_lon_deg * (1 - overlap)
        
        # Calculate number of tiles needed
        lat_span = bbox['max_lat'] - bbox['min_lat']
        lon_span = bbox['max_lon'] - bbox['min_lon']
        
        num_rows = max(1, int(np.ceil(lat_span / step_lat)))
        num_cols = max(1, int(np.ceil(lon_span / step_lon)))
        
        print(f"Tile grid: {num_rows} rows × {num_cols} cols = {num_rows * num_cols} tiles")
        print(f"Tile size: {tile_size}x{tile_size}@2x (zoom {tile_zoom})")
        print(f"Overlap: {overlap*100:.0f}%")
        
        # Create tiles directory
        tiles_dir = output_dir / f"area_{area_id}_tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Download tiles
        tiles = []
        total = num_rows * num_cols
        
        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate tile center
                center_lat_tile = bbox['min_lat'] + (row + 0.5) * step_lat
                center_lon_tile = bbox['min_lon'] + (col + 0.5) * step_lon
                
                # Clamp to bounding box
                center_lat_tile = max(bbox['min_lat'], min(bbox['max_lat'], center_lat_tile))
                center_lon_tile = max(bbox['min_lon'], min(bbox['max_lon'], center_lon_tile))
                
                tile_filename = f"tile_r{row}_c{col}.png"
                tile_path = tiles_dir / tile_filename
                
                # Download tile
                print(f"  Downloading tile {row}x{col} ({len(tiles)+1}/{total})...", end=' ')
                
                if tile_path.exists():
                    print("(cached)")
                else:
                    success = self._download_tile(
                        center_lat_tile, center_lon_tile,
                        tile_zoom, tile_size, tile_path
                    )
                    
                    if success:
                        print("✓")
                    else:
                        print("✗ failed")
                        continue
                    
                    time.sleep(0.3)  # Rate limiting
                
                tiles.append({
                    'path': tile_path,
                    'row': row,
                    'col': col,
                    'lat': center_lat_tile,
                    'lon': center_lon_tile,
                    'zoom': tile_zoom
                })
        
        print(f"\n✓ Downloaded {len(tiles)}/{total} tiles")
        return tiles
    
    def _download_tile(self, lat: float, lon: float, zoom: int, 
                      size: int, output_path: Path) -> bool:
        """Download a single tile from Google Maps Static API."""
        url = "https://maps.googleapis.com/maps/api/staticmap"
        
        params = {
            'center': f'{lat},{lon}',
            'zoom': zoom,
            'size': f'{size}x{size}',
            'scale': 2,
            'maptype': 'satellite',
            'format': 'png',
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('application/json'):
                return False
            
            img = Image.open(BytesIO(response.content))
            img.save(output_path)
            return True
            
        except Exception as e:
            return False
    
    def stage3_detect_vehicles(self, tiles: List[Dict], 
                               conf_threshold: float = 0.25) -> List[Dict]:
        """
        Stage 3: Run vehicle/stall detection on all tiles.
        Returns detection results for each tile with class separation.
        """
        print("\n" + "="*70)
        print("STAGE 3: Vehicle & Stall Detection")
        print("="*70)
        
        # Class names: 0=car, 1=lot_boundary, 2=objects, 3=stall
        class_names = ['car', 'lot_boundary', 'objects', 'stall']
        
        results = []
        total_cars = 0
        total_stalls = 0
        total_objects = 0
        
        for idx, tile in enumerate(tiles):
            print(f"\nProcessing tile {idx+1}/{len(tiles)}: {tile['path'].name}")
            
            # Run detection
            detections = self.detection_model.predict(
                source=str(tile['path']),
                conf=conf_threshold,
                iou=0.45,
                verbose=False,
                device='mps'
            )[0]
            
            # Separate detections by class
            cars = []
            stalls = []
            objects = []
            
            if detections.boxes is not None:
                for box in detections.boxes:
                    cls = int(box.cls.item())
                    
                    if cls == 0:  # car
                        cars.append(box)
                    elif cls == 3:  # stall
                        stalls.append(box)
                    elif cls == 2:  # objects
                        objects.append(box)
            
            num_cars = len(cars)
            num_stalls = len(stalls)
            num_objects = len(objects)
            
            print(f"  Cars: {num_cars} | Stalls: {num_stalls} | Objects: {num_objects}")
            
            total_cars += num_cars
            total_stalls += num_stalls
            total_objects += num_objects
            
            # Store results
            tile_result = {
                'tile': tile,
                'detections': detections,
                'cars': cars,
                'stalls': stalls,
                'objects': objects,
                'num_cars': num_cars,
                'num_stalls': num_stalls,
                'num_objects': num_objects
            }
            
            results.append(tile_result)
        
        print(f"\n✓ Total across all tiles:")
        print(f"  Cars: {total_cars}")
        print(f"  Stalls: {total_stalls}")
        print(f"  Objects: {total_objects}")
        
        return results
    
    def _calculate_iou(self, box1, box2) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _match_cars_to_stalls(self, cars_global, stalls_global, iou_threshold=0.3):
        """Match detected cars to parking stalls based on IoU overlap."""
        occupied_stalls = []
        empty_stalls = list(range(len(stalls_global)))
        unmatched_cars = []
        
        for car_idx, car_box in enumerate(cars_global):
            best_iou = 0
            best_stall_idx = -1
            
            for stall_idx in empty_stalls:
                stall_box = stalls_global[stall_idx]
                iou = self._calculate_iou(car_box[:4], stall_box[:4])
                
                if iou > best_iou:
                    best_iou = iou
                    best_stall_idx = stall_idx
            
            if best_iou >= iou_threshold:
                occupied_stalls.append({
                    'stall_idx': best_stall_idx,
                    'car_idx': car_idx,
                    'iou': best_iou,
                    'car_box': car_box,
                    'stall_box': stalls_global[best_stall_idx]
                })
                empty_stalls.remove(best_stall_idx)
            else:
                unmatched_cars.append(car_idx)
        
        return occupied_stalls, empty_stalls, unmatched_cars
    
    def stage4_advanced_occupancy_analysis(self, parking_area: Dict, 
                                           tile_results: List[Dict],
                                           output_dir: Path) -> Dict:
        """
        Stage 4: Advanced occupancy analysis with stall matching and visualization.
        Returns comprehensive occupancy statistics.
        """
        print("\n" + "="*70)
        print("STAGE 4: Advanced Occupancy Analysis")
        print("="*70)
        
        if not tile_results:
            print("No results to analyze")
            return {}
        
        # Determine grid layout
        rows = max(tr['tile']['row'] for tr in tile_results) + 1
        cols = max(tr['tile']['col'] for tr in tile_results) + 1
        
        print(f"Grid layout: {rows} rows × {cols} cols")
        
        # Load first tile to get dimensions
        first_tile = cv2.imread(str(tile_results[0]['tile']['path']))
        tile_h, tile_w = first_tile.shape[:2]
        
        # Create stitched canvas
        canvas_h = tile_h * rows
        canvas_w = tile_w * cols
        stitched = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Overlay tiles
        print("\nStitching tiles...")
        for tr in tile_results:
            tile_img = cv2.imread(str(tr['tile']['path']))
            row = tr['tile']['row']
            col = tr['tile']['col']
            
            y1 = row * tile_h
            x1 = col * tile_w
            
            stitched[y1:y1+tile_h, x1:x1+tile_w] = tile_img
        
        # Collect all detections in global coordinates
        print("\nCollecting detections...")
        cars_global = []
        stalls_global = []
        objects_global = []
        
        for tr in tile_results:
            row = tr['tile']['row']
            col = tr['tile']['col']
            offset_y = row * tile_h
            offset_x = col * tile_w
            
            # Process cars
            for box in tr['cars']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                cars_global.append([
                    int(x1 + offset_x), int(y1 + offset_y),
                    int(x2 + offset_x), int(y2 + offset_y),
                    conf
                ])
            
            # Process stalls
            for box in tr['stalls']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                stalls_global.append([
                    int(x1 + offset_x), int(y1 + offset_y),
                    int(x2 + offset_x), int(y2 + offset_y),
                    conf
                ])
            
            # Process objects
            for box in tr['objects']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                objects_global.append([
                    int(x1 + offset_x), int(y1 + offset_y),
                    int(x2 + offset_x), int(y2 + offset_y),
                    conf
                ])
        
        num_cars = len(cars_global)
        num_stalls = len(stalls_global)
        num_objects = len(objects_global)
        
        print(f"  Total cars: {num_cars}")
        print(f"  Total stalls: {num_stalls}")
        print(f"  Total objects: {num_objects}")
        
        # Match cars to stalls
        print("\nMatching cars to stalls...")
        occupied_stalls, empty_stalls, unmatched_cars = self._match_cars_to_stalls(
            cars_global, stalls_global
        )
        
        num_occupied = len(occupied_stalls)
        num_empty = len(empty_stalls)
        occupancy_rate = (num_occupied / num_stalls * 100) if num_stalls > 0 else 0
        
        print(f"  Occupied stalls: {num_occupied}")
        print(f"  Empty stalls: {num_empty}")
        print(f"  Unmatched cars: {len(unmatched_cars)} (cars not in stalls)")
        print(f"  Occupancy rate: {occupancy_rate:.1f}%")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # 1. Standard detection visualization
        vis_standard = stitched.copy()
        
        # Draw empty stalls (blue)
        for stall_idx in empty_stalls:
            x1, y1, x2, y2, conf = stalls_global[stall_idx]
            cv2.rectangle(vis_standard, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
        
        # Draw occupied stalls (green) and their cars (red)
        for match in occupied_stalls:
            stall = match['stall_box']
            car = match['car_box']
            
            # Stall in green
            cv2.rectangle(vis_standard, (stall[0], stall[1]), (stall[2], stall[3]), 
                         (0, 255, 0), 2)
            # Car in red
            cv2.rectangle(vis_standard, (car[0], car[1]), (car[2], car[3]), 
                         (0, 0, 255), 2)
        
        # Draw unmatched cars (yellow)
        for car_idx in unmatched_cars:
            car = cars_global[car_idx]
            cv2.rectangle(vis_standard, (car[0], car[1]), (car[2], car[3]), 
                         (0, 255, 255), 3)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_standard, "Empty Stalls (Blue)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis_standard, "Occupied Stalls (Green)", (10, legend_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_standard, "Cars (Red)", (10, legend_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_standard, "Unmatched Cars (Yellow)", (10, legend_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_standard, f"Occupancy: {occupancy_rate:.1f}%", (10, legend_y + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Save standard visualization
        output_standard = output_dir / f"area_{parking_area['id']}_occupancy.jpg"
        cv2.imwrite(str(output_standard), vis_standard)
        print(f"  ✓ Occupancy visualization: {output_standard.name}")
        
        # 2. Heat map visualization
        vis_heatmap = self._create_occupancy_heatmap(
            stitched, stalls_global, occupied_stalls, empty_stalls
        )
        output_heatmap = output_dir / f"area_{parking_area['id']}_heatmap.jpg"
        cv2.imwrite(str(output_heatmap), vis_heatmap)
        print(f"  ✓ Heat map: {output_heatmap.name}")
        
        # 3. Export detailed data
        self._export_occupancy_data(
            parking_area, cars_global, stalls_global, occupied_stalls, 
            empty_stalls, unmatched_cars, output_dir
        )
        
        # Calculate comprehensive statistics
        stats = {
            'parking_area_id': parking_area['id'],
            'dimensions': {
                'width_m': parking_area['size_meters']['width'],
                'height_m': parking_area['size_meters']['height']
            },
            'detections': {
                'total_cars': num_cars,
                'total_stalls': num_stalls,
                'total_objects': num_objects
            },
            'occupancy': {
                'occupied_stalls': num_occupied,
                'empty_stalls': num_empty,
                'unmatched_cars': len(unmatched_cars),
                'occupancy_rate': round(occupancy_rate, 2)
            },
            'visualizations': {
                'standard': str(output_standard),
                'heatmap': str(output_heatmap)
            }
        }
        
        print(f"\n✓ Analysis complete for area #{parking_area['id']}")
        
        return stats
    
    def _create_occupancy_heatmap(self, base_image, stalls_global, 
                                  occupied_stalls, empty_stalls):
        """Create heat map visualization of parking occupancy."""
        heatmap = base_image.copy()
        overlay = np.zeros_like(heatmap, dtype=np.uint8)
        
        # Draw empty stalls with blue tint
        for stall_idx in empty_stalls:
            x1, y1, x2, y2, _ = stalls_global[stall_idx]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 0), -1)
        
        # Draw occupied stalls with red tint
        for match in occupied_stalls:
            stall = match['stall_box']
            cv2.rectangle(overlay, (stall[0], stall[1]), (stall[2], stall[3]), 
                         (0, 100, 255), -1)
        
        # Blend overlay with base image
        heatmap = cv2.addWeighted(heatmap, 0.7, overlay, 0.3, 0)
        
        # Add text overlay
        cv2.putText(heatmap, "Occupancy Heat Map", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(heatmap, "Blue = Empty | Red = Occupied", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return heatmap
    
    def _export_occupancy_data(self, parking_area, cars_global, stalls_global,
                               occupied_stalls, empty_stalls, unmatched_cars, 
                               output_dir):
        """Export detailed occupancy data to JSON and CSV formats."""
        area_id = parking_area['id']
        
        # JSON export with full details
        json_data = {
            'parking_area_id': area_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dimensions': {
                'width_m': float(parking_area['size_meters']['width']),
                'height_m': float(parking_area['size_meters']['height'])
            },
            'summary': {
                'total_stalls': len(stalls_global),
                'occupied_stalls': len(occupied_stalls),
                'empty_stalls': len(empty_stalls),
                'total_cars': len(cars_global),
                'unmatched_cars': len(unmatched_cars),
                'occupancy_rate': round(len(occupied_stalls) / len(stalls_global) * 100, 2) if stalls_global else 0
            },
            'stalls': [],
            'unmatched_cars': []
        }
        
        # Add empty stalls
        for stall_idx in empty_stalls:
            stall = stalls_global[stall_idx]
            json_data['stalls'].append({
                'stall_id': int(stall_idx),
                'status': 'empty',
                'bbox': [int(stall[0]), int(stall[1]), int(stall[2]), int(stall[3])],
                'confidence': float(stall[4])
            })
        
        # Add occupied stalls
        for match in occupied_stalls:
            stall = match['stall_box']
            car = match['car_box']
            json_data['stalls'].append({
                'stall_id': int(match['stall_idx']),
                'status': 'occupied',
                'bbox': [int(stall[0]), int(stall[1]), int(stall[2]), int(stall[3])],
                'confidence': float(stall[4]),
                'car_bbox': [int(car[0]), int(car[1]), int(car[2]), int(car[3])],
                'car_confidence': float(car[4]),
                'match_iou': float(match['iou'])
            })
        
        # Add unmatched cars
        for car_idx in unmatched_cars:
            car = cars_global[car_idx]
            json_data['unmatched_cars'].append({
                'car_id': int(car_idx),
                'bbox': [int(car[0]), int(car[1]), int(car[2]), int(car[3])],
                'confidence': float(car[4])
            })
        
        # Save JSON
        json_path = output_dir / f"area_{area_id}_occupancy_data.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"  ✓ Data export (JSON): {json_path.name}")
        
        # CSV export for easy analysis
        import csv
        csv_path = output_dir / f"area_{area_id}_occupancy_data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Stall ID', 'Status', 'X1', 'Y1', 'X2', 'Y2', 
                           'Stall Conf', 'Car X1', 'Car Y1', 'Car X2', 'Car Y2', 
                           'Car Conf', 'Match IoU'])
            
            # Empty stalls
            for stall_idx in empty_stalls:
                stall = stalls_global[stall_idx]
                writer.writerow([stall_idx, 'empty', stall[0], stall[1], 
                               stall[2], stall[3], round(stall[4], 3),
                               '', '', '', '', '', ''])
            
            # Occupied stalls
            for match in occupied_stalls:
                stall = match['stall_box']
                car = match['car_box']
                writer.writerow([match['stall_idx'], 'occupied', 
                               stall[0], stall[1], stall[2], stall[3], 
                               round(stall[4], 3),
                               car[0], car[1], car[2], car[3], 
                               round(car[4], 3), round(match['iou'], 3)])
        
        print(f"  ✓ Data export (CSV): {csv_path.name}")
    
    def run_full_pipeline(self, wide_area_image: Path,
                         center_lat: float, center_lon: float,
                         zoom: int = 19,
                         output_dir: Path = None,
                         conf_stage1: float = 0.6,
                         conf_stage3: float = 0.25) -> Dict:
        """Run complete detection pipeline."""
        
        if output_dir is None:
            output_dir = wide_area_image.parent / 'pipeline_results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("MULTI-STAGE PARKING DETECTION PIPELINE")
        print("="*70)
        
        # Stage 1: Detect parking areas
        parking_areas = self.stage1_detect_parking_areas(
            wide_area_image, center_lat, center_lon, zoom, conf_stage1
        )
        
        if not parking_areas:
            print("\n❌ No parking areas detected")
            return {'parking_areas': [], 'results': []}
        
        # Collect all tiles from all parking areas
        print("\n" + "="*70)
        print("STAGE 2: Tile Download (All Parking Areas)")
        print("="*70)
        
        all_tiles = []
        for area in parking_areas:
            tiles = self.stage2_plan_and_download_tiles(area, output_dir)
            if tiles:
                all_tiles.extend(tiles)
        
        if not all_tiles:
            print("\n❌ No tiles downloaded")
            return {'parking_areas': [], 'results': {}}
        
        print(f"\n✓ Total tiles collected: {len(all_tiles)}")
        
        # Stage 3: Detect vehicles on all tiles
        tile_results = self.stage3_detect_vehicles(all_tiles, conf_stage3)
        
        # Stage 4: Create unified occupancy analysis
        print("\n" + "="*70)
        print("STAGE 4: Overall Occupancy Analysis")
        print("="*70)
        
        # Load original wide-area image for reference
        wide_img = cv2.imread(str(wide_area_image))
        img_h, img_w = wide_img.shape[:2]
        
        # Determine the bounding box of all tiles to create stitched canvas
        print("\nCalculating canvas dimensions...")
        
        # Find extent of all parking areas in pixel coordinates
        min_x = min(area['bbox_pixels']['x1'] for area in parking_areas)
        min_y = min(area['bbox_pixels']['y1'] for area in parking_areas)
        max_x = max(area['bbox_pixels']['x2'] for area in parking_areas)
        max_y = max(area['bbox_pixels']['y2'] for area in parking_areas)
        
        # Calculate scale factor (zoom 19 to zoom 20 = 2x)
        scale = 2
        
        # Create a large canvas for stitching all tiles
        canvas_w = (max_x - min_x) * scale
        canvas_h = (max_y - min_y) * scale
        stitched_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        print(f"  Canvas size: {canvas_w}x{canvas_h} pixels")
        
        # Stitch all tiles onto the canvas
        print("\nStitching all tiles...")
        tile_h, tile_w = 1280, 1280  # 640x640@2x
        
        for tr in tile_results:
            tile_img = cv2.imread(str(tr['tile']['path']))
            
            # Get tile's position in global coordinates
            tile_center_lat = tr['tile']['lat']
            tile_center_lon = tr['tile']['lon']
            
            # Convert to pixel coordinates in original image
            tile_x_center = img_w / 2 + (tile_center_lon - center_lon) / (360 / (256 * (2 ** zoom)))
            tile_y_center = img_h / 2 - (tile_center_lat - center_lat) / (360 / (256 * (2 ** zoom)))
            
            # Scale to stitched canvas coordinates
            canvas_x = int((tile_x_center - min_x) * scale - tile_w / 2)
            canvas_y = int((tile_y_center - min_y) * scale - tile_h / 2)
            
            # Clip to canvas bounds
            x1 = max(0, canvas_x)
            y1 = max(0, canvas_y)
            x2 = min(canvas_w, canvas_x + tile_w)
            y2 = min(canvas_h, canvas_y + tile_h)
            
            # Calculate source coordinates
            src_x1 = x1 - canvas_x
            src_y1 = y1 - canvas_y
            src_x2 = src_x1 + (x2 - x1)
            src_y2 = src_y1 + (y2 - y1)
            
            if x2 > x1 and y2 > y1:
                stitched_canvas[y1:y2, x1:x2] = tile_img[src_y1:src_y2, src_x1:src_x2]
        
        # Collect all detections in canvas coordinates
        print("\nCollecting all detections...")
        all_cars = []
        all_stalls = []
        all_objects = []
        
        for tr in tile_results:
            tile_center_lat = tr['tile']['lat']
            tile_center_lon = tr['tile']['lon']
            
            # Convert to pixel coordinates
            tile_x_center = img_w / 2 + (tile_center_lon - center_lon) / (360 / (256 * (2 ** zoom)))
            tile_y_center = img_h / 2 - (tile_center_lat - center_lat) / (360 / (256 * (2 ** zoom)))
            
            canvas_x = int((tile_x_center - min_x) * scale - tile_w / 2)
            canvas_y = int((tile_y_center - min_y) * scale - tile_h / 2)
            
            # Process cars
            for box in tr['cars']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                all_cars.append([
                    int(x1 + canvas_x), int(y1 + canvas_y),
                    int(x2 + canvas_x), int(y2 + canvas_y),
                    conf
                ])
            
            # Process stalls
            for box in tr['stalls']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                all_stalls.append([
                    int(x1 + canvas_x), int(y1 + canvas_y),
                    int(x2 + canvas_x), int(y2 + canvas_y),
                    conf
                ])
            
            # Process objects
            for box in tr['objects']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                all_objects.append([
                    int(x1 + canvas_x), int(y1 + canvas_y),
                    int(x2 + canvas_x), int(y2 + canvas_y),
                    conf
                ])
        
        total_cars = len(all_cars)
        total_stalls = len(all_stalls)
        total_objects = len(all_objects)
        
        print(f"  Total cars: {total_cars}")
        print(f"  Total stalls: {total_stalls}")
        print(f"  Total objects: {total_objects}")
        
        # Match cars to stalls
        print("\nMatching cars to stalls...")
        occupied_stalls, empty_stalls, unmatched_cars = self._match_cars_to_stalls(
            all_cars, all_stalls
        )
        
        num_occupied = len(occupied_stalls)
        num_empty = len(empty_stalls)
        occupancy_rate = (num_occupied / total_stalls * 100) if total_stalls > 0 else 0
        
        print(f"  Occupied stalls: {num_occupied}")
        print(f"  Empty stalls: {num_empty}")
        print(f"  Unmatched cars: {len(unmatched_cars)} (not in stalls)")
        print(f"\n  🅿️  OVERALL OCCUPANCY: {occupancy_rate:.1f}% ({num_occupied}/{total_stalls} stalls)")
        
        # Draw all detections on stitched canvas
        print("\nGenerating overall visualization...")
        vis_canvas = stitched_canvas.copy()
        
        # Draw empty stalls (blue)
        for stall_idx in empty_stalls:
            x1, y1, x2, y2, conf = all_stalls[stall_idx]
            cv2.rectangle(vis_canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw occupied stalls (green) and their cars (red)
        for match in occupied_stalls:
            stall = match['stall_box']
            car = match['car_box']
            
            # Stall in green
            cv2.rectangle(vis_canvas, (stall[0], stall[1]), (stall[2], stall[3]), 
                         (0, 255, 0), 2)
            # Car in red
            cv2.rectangle(vis_canvas, (car[0], car[1]), (car[2], car[3]), 
                         (0, 0, 255), 2)
        
        # Draw unmatched cars (yellow)
        for car_idx in unmatched_cars:
            car = all_cars[car_idx]
            cv2.rectangle(vis_canvas, (car[0], car[1]), (car[2], car[3]), 
                         (0, 255, 255), 3)
        
        # Add legend and statistics
        legend_y = 40
        line_height = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        # Background for text
        text_bg_h = 250
        text_bg_w = min(700, canvas_w - 20)
        cv2.rectangle(vis_canvas, (10, 10), (10 + text_bg_w, 10 + text_bg_h), (0, 0, 0), -1)
        cv2.rectangle(vis_canvas, (10, 10), (10 + text_bg_w, 10 + text_bg_h), (255, 255, 255), 3)
        
        cv2.putText(vis_canvas, "PARKING LOT OCCUPANCY", (20, legend_y), 
                   font, font_scale, (255, 255, 255), thickness)
        legend_y += line_height
        
        cv2.putText(vis_canvas, f"Total Stalls: {total_stalls}", (20, legend_y), 
                   font, font_scale, (255, 255, 255), thickness)
        legend_y += line_height
        
        cv2.putText(vis_canvas, f"Occupied: {num_occupied} | Empty: {num_empty}", (20, legend_y), 
                   font, font_scale, (0, 255, 0) if num_occupied < total_stalls else (0, 165, 255), thickness)
        legend_y += line_height
        
        cv2.putText(vis_canvas, f"Occupancy: {occupancy_rate:.1f}%", (20, legend_y), 
                   font, 1.4, (0, 255, 0) if occupancy_rate < 80 else (0, 165, 255), 4)
        legend_y += line_height
        
        # Color legend
        cv2.rectangle(vis_canvas, (20, legend_y), (50, legend_y + 20), (255, 0, 0), -1)
        cv2.putText(vis_canvas, "Empty", (60, legend_y + 18), font, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(vis_canvas, (200, legend_y), (230, legend_y + 20), (0, 255, 0), -1)
        cv2.putText(vis_canvas, "Occupied", (240, legend_y + 18), font, 0.7, (255, 255, 255), 2)
        
        # Save overall visualization
        output_path = output_dir / 'overall_occupancy.jpg'
        cv2.imwrite(str(output_path), vis_canvas)
        print(f"  ✓ Overall visualization: {output_path.name}")
        
        # Save detailed data
        overall_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'location': {
                'latitude': center_lat,
                'longitude': center_lon
            },
            'summary': {
                'total_parking_areas': len(parking_areas),
                'total_stalls': total_stalls,
                'occupied_stalls': num_occupied,
                'empty_stalls': num_empty,
                'total_cars': total_cars,
                'unmatched_cars': len(unmatched_cars),
                'occupancy_rate': round(occupancy_rate, 2)
            }
        }
        
        json_path = output_dir / 'overall_occupancy.json'
        with open(json_path, 'w') as f:
            json.dump(overall_data, f, indent=2)
        print(f"  ✓ Overall data: {json_path.name}")
        
        # Clean up individual area files
        print("\nCleaning up intermediate files...")
        for i in range(1, len(parking_areas) + 1):
            # Remove individual area visualizations
            for pattern in [f'area_{i}_occupancy.jpg', f'area_{i}_heatmap.jpg', 
                          f'area_{i}_occupancy_data.json', f'area_{i}_occupancy_data.csv',
                          f'area_{i}_stitched.jpg']:
                file_path = output_dir / pattern
                if file_path.exists():
                    file_path.unlink()
            
            # Remove tile directories
            tile_dir = output_dir / f'area_{i}_tiles'
            if tile_dir.exists():
                import shutil
                shutil.rmtree(tile_dir)
        
        print("  ✓ Cleaned up intermediate files")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Overall Occupancy: {occupancy_rate:.1f}% ({num_occupied}/{total_stalls} stalls)")
        print(f"Output: {output_path}")
        
        return overall_data


def main():
    parser = argparse.ArgumentParser(
        description="Multi-stage parking lot detection pipeline"
    )
    parser.add_argument('--image', type=str, required=True,
                       help='Path to wide-area satellite image (zoom 19)')
    parser.add_argument('--lat', type=float, required=True,
                       help='Latitude of image center')
    parser.add_argument('--lon', type=float, required=True,
                       help='Longitude of image center')
    parser.add_argument('--zoom', type=int, default=19,
                       help='Zoom level of input image (default: 19)')
    parser.add_argument('--output', type=str,
                       help='Output directory for results')
    parser.add_argument('--conf-stage1', type=float, default=0.6,
                       help='Confidence threshold for parking lot detection (default: 0.6)')
    parser.add_argument('--conf-stage3', type=float, default=0.25,
                       help='Confidence threshold for vehicle detection (default: 0.25)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ParkingDetectionPipeline()
    
    # Run pipeline
    image_path = Path(args.image)
    output_dir = Path(args.output) if args.output else None
    
    results = pipeline.run_full_pipeline(
        image_path,
        args.lat,
        args.lon,
        args.zoom,
        output_dir,
        args.conf_stage1,
        args.conf_stage3
    )
    
    print("\n✓ Pipeline execution complete!")


if __name__ == '__main__':
    main()
