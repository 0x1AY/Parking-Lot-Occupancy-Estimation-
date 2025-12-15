#!/usr/bin/env python3
"""
Unified Parking Occupancy Detection Pipeline
============================================
Complete end-to-end pipeline that:
1. Detects parking areas from wide satellite image (zoom 19)
2. Downloads high-res tiles covering ALL parking areas (zoom 20)
3. Detects cars and stalls on all tiles
4. Stitches everything into one coherent image
5. Calculates overall occupancy for the entire parking lot

Usage:
    python occupancy/unified_parking_pipeline.py \
        --image walmart_locations/wide_area_z19/walmart_01_*.png \
        --lat 43.668734 --lon -79.340158
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
import math


class UnifiedParkingPipeline:
    """Unified parking occupancy detection with dual-model approach."""
    
    def __init__(self, 
                 localization_model_path: str = "datasets/apklot/apklot_stage1/weights/best.pt",
                 car_model_path: str = "parking_runs/yolo11m_parking_augmented2/weights/best.pt",
                 stall_model_path: str = "parking_runs/yolo11m_multilabel/weights/best.pt",
                 google_api_key: str = "AIzaSyCZWUlRCSb7WxHNBWtMifWRW25GOWfbous"):
        """
        Initialize pipeline with dual detection models.
        
        Args:
            localization_model_path: Path to parking lot localization model
            car_model_path: Path to high-accuracy car detection model (96.3% mAP50)
            stall_model_path: Path to multiclass model for stall detection
            google_api_key: Google Maps API key
        """
        self.api_key = google_api_key
        
        print(f"Loading localization model: {localization_model_path}")
        self.localization_model = YOLO(localization_model_path)
        print("✓ Localization model loaded")
        
        print(f"Loading car detection model: {car_model_path}")
        self.car_model = YOLO(car_model_path)
        print("✓ Car detection model loaded (high-accuracy)")
        
        print(f"Loading stall detection model: {stall_model_path}")
        self.stall_model = YOLO(stall_model_path)
        print("✓ Stall detection model loaded")
    
    def download_satellite_image(self, lat: float, lon: float, 
                                 zoom: int, size: int, 
                                 output_path: Path) -> Path:
        """Download satellite image from Google Static Maps API."""
        url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}&zoom={zoom}&size={size}x{size}"
            f"&maptype=satellite&scale=2&key={self.api_key}"
        )
        
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(output_path)
            return output_path
        else:
            error_msg = f"Failed to download image: HTTP {response.status_code}"
            if response.status_code == 403:
                error_msg += (
                    "\n\nPossible causes:"
                    "\n1. API key is invalid or expired"
                    "\n2. Static Maps API is not enabled in Google Cloud Console"
                    "\n3. API key doesn't have permission for Static Maps API"
                    "\n4. Billing is not enabled on your Google Cloud project"
                    "\n\nTo fix:"
                    "\n1. Go to: https://console.cloud.google.com/google/maps-apis"
                    "\n2. Enable 'Maps Static API'"
                    "\n3. Check that billing is enabled"
                    "\n4. Verify API key restrictions allow this API"
                )
            elif response.status_code == 400:
                error_msg += f"\n\nInvalid request parameters. Response: {response.text[:200]}"
            raise Exception(error_msg)
    
    def process_location(self, location_name: str, lat: float, lon: float,
                        output_dir: Path,
                        localization_zoom: int = 19,
                        tile_zoom: int = 20,
                        conf_threshold: float = 0.25,
                        iou_threshold: float = 0.3) -> Dict:
        """
        Process a parking location from scratch given lat/lon coordinates.
        
        Args:
            location_name: Name for this location
            lat: Latitude of parking lot center
            lon: Longitude of parking lot center
            output_dir: Directory to save results
            localization_zoom: Zoom level for initial parking detection
            tile_zoom: Zoom level for high-res tile downloads
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for car-to-stall matching
            
        Returns:
            Dictionary with results including occupancy metrics and paths
        """
        location_dir = output_dir / location_name
        location_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print(f"PROCESSING LOCATION: {location_name}")
        print("="*70)
        print(f"Coordinates: {lat:.6f}, {lon:.6f}")
        print(f"Output: {location_dir}")
        
        # Download wide area image for localization
        print(f"\n Downloading wide area image (zoom {localization_zoom})...")
        wide_img_path = location_dir / f"{location_name}_z{localization_zoom}.png"
        self.download_satellite_image(lat, lon, localization_zoom, 640, wide_img_path)
        print(f"✓ Downloaded: {wide_img_path.name}")
        
        # Run the pipeline
        results = self.run_pipeline(
            wide_area_image=wide_img_path,
            center_lat=lat,
            center_lon=lon,
            zoom=localization_zoom,
            output_dir=location_dir,
            conf_stage1=0.6,
            conf_stage3=conf_threshold
        )
        
        if results is None:
            return {
                'location_name': location_name,
                'latitude': lat,
                'longitude': lon,
                'total_stalls': 0,
                'occupied_stalls': 0,
                'occupancy_rate': 0.0,
                'cars_detected': 0,
                'result_path': None,
                'error': 'No parking areas detected'
            }
        
        # Format results for app
        summary = results.get('summary', {})
        result_img_path = location_dir / 'overall_occupancy.jpg'
        
        return {
            'location_name': location_name,
            'latitude': lat,
            'longitude': lon,
            'total_stalls': summary.get('total_stalls', 0),
            'occupied_stalls': summary.get('occupied_stalls', 0),
            'empty_stalls': summary.get('empty_stalls', 0),
            'occupancy_rate': summary.get('occupancy_rate', 0.0),
            'cars_detected': summary.get('total_cars', 0),
            'unmatched_cars': summary.get('unmatched_cars', 0),
            'result_path': str(result_img_path) if result_img_path.exists() else None,
            'timestamp': results.get('timestamp', ''),
            'processing_success': True
        }
    
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
                                     zoom: int = 19, conf_threshold: float = 0.6) -> Dict:
        """
        Stage 1: Detect all parking lot areas and calculate bounding box.
        Returns combined bounding box covering all parking areas.
        """
        print("\n" + "="*70)
        print("STAGE 1: Parking Lot Localization")
        print("="*70)
        
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
        print(f"✓ Found {num_detections} parking area(s)")
        
        if num_detections == 0:
            return None
        
        # Find combined bounding box covering all parking areas
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        parking_areas = []
        for idx, box in enumerate(results.boxes):
            conf = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Update combined bounds
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            
            parking_areas.append({
                'id': idx + 1,
                'confidence': float(conf),
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
        
        # Convert combined bounding box to lat/lon
        lat_min, lon_min = self.pixel_to_latlon(
            min_x, max_y, img_width, img_height, center_lat, center_lon, zoom
        )
        lat_max, lon_max = self.pixel_to_latlon(
            max_x, min_y, img_width, img_height, center_lat, center_lon, zoom
        )
        
        # Calculate dimensions
        width_m = (lon_max - lon_min) * 111319.9 * math.cos(math.radians(center_lat))
        height_m = (lat_max - lat_min) * 111319.9
        
        combined_bbox = {
            'pixel_bounds': {
                'x1': int(min_x), 'y1': int(min_y),
                'x2': int(max_x), 'y2': int(max_y)
            },
            'geo_bounds': {
                'min_lat': lat_min, 'max_lat': lat_max,
                'min_lon': lon_min, 'max_lon': lon_max,
                'center_lat': (lat_min + lat_max) / 2,
                'center_lon': (lon_min + lon_max) / 2
            },
            'dimensions': {
                'width_m': abs(width_m),
                'height_m': abs(height_m)
            },
            'parking_areas': parking_areas
        }
        
        print(f"\nCombined parking lot coverage:")
        print(f"  Size: {abs(width_m):.1f}m × {abs(height_m):.1f}m")
        print(f"  Bounds: ({lat_min:.6f}, {lon_min:.6f}) to ({lat_max:.6f}, {lon_max:.6f})")
        print(f"  Total parking areas: {len(parking_areas)}")
        
        return combined_bbox
    
    def stage2_download_tiles(self, combined_bbox: Dict, output_dir: Path,
                              tile_zoom: int = 20, tile_size: int = 640,
                              overlap: float = 0.2) -> List[Dict]:
        """
        Stage 2: Download tiles covering the entire combined parking area.
        Returns list of tiles with their grid positions.
        """
        print("\n" + "="*70)
        print("STAGE 2: High-Resolution Tile Download")
        print("="*70)
        
        bbox = combined_bbox['geo_bounds']
        center_lat = bbox['center_lat']
        
        # Calculate tile coverage
        meters_per_pixel = self.calculate_meters_per_pixel(tile_zoom, center_lat)
        tile_meters = tile_size * 2 * meters_per_pixel  # scale=2 for retina
        tile_lat_deg = tile_meters / 111319.9
        tile_lon_deg = tile_meters / (111319.9 * math.cos(math.radians(center_lat)))
        
        # Calculate step size with overlap
        step_lat = tile_lat_deg * (1 - overlap)
        step_lon = tile_lon_deg * (1 - overlap)
        
        # Calculate number of tiles
        lat_span = bbox['max_lat'] - bbox['min_lat']
        lon_span = bbox['max_lon'] - bbox['min_lon']
        
        num_rows = max(1, int(np.ceil(lat_span / step_lat)))
        num_cols = max(1, int(np.ceil(lon_span / step_lon)))
        
        print(f"Tile grid: {num_rows} rows × {num_cols} cols = {num_rows * num_cols} tiles")
        print(f"Tile size: {tile_size}x{tile_size}@2x (zoom {tile_zoom})")
        print(f"Coverage: {combined_bbox['dimensions']['width_m']:.1f}m × {combined_bbox['dimensions']['height_m']:.1f}m")
        print(f"Overlap: {overlap*100:.0f}%")
        
        # Create tiles directory
        tiles_dir = output_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Download tiles
        tiles = []
        total = num_rows * num_cols
        
        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate tile center
                center_lat_tile = bbox['min_lat'] + (row + 0.5) * step_lat
                center_lon_tile = bbox['min_lon'] + (col + 0.5) * step_lon
                
                tile_filename = f"tile_r{row}_c{col}.png"
                tile_path = tiles_dir / tile_filename
                
                print(f"  Tile [{row},{col}] ({len(tiles)+1}/{total})...", end=' ')
                
                if tile_path.exists():
                    print("cached")
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
                    
                    time.sleep(0.3)
                
                tiles.append({
                    'path': tile_path,
                    'row': row,
                    'col': col,
                    'lat': center_lat_tile,
                    'lon': center_lon_tile,
                    'zoom': tile_zoom
                })
        
        print(f"\n✓ Downloaded {len(tiles)}/{total} tiles")
        return tiles, num_rows, num_cols
    
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
            
        except Exception:
            return False
    
    def stage3_detect_objects(self, tiles: List[Dict], 
                              conf_threshold: float = 0.25) -> List[Dict]:
        """
        Stage 3: Run dual-model detection on all tiles.
        Uses high-accuracy car model (96.3% mAP50) for cars and multiclass model for stalls.
        Returns detection results for each tile.
        """
        print("\n" + "="*70)
        print("STAGE 3: Dual-Model Vehicle & Stall Detection")
        print("="*70)
        
        results = []
        total_cars = 0
        total_stalls = 0
        
        for idx, tile in enumerate(tiles):
            print(f"  [{tile['row']},{tile['col']}] {idx+1}/{len(tiles)}...", end=' ')
            
            # Run car detection with high-accuracy model
            car_detections = self.car_model.predict(
                source=str(tile['path']),
                classes=[0],  # car class only
                conf=conf_threshold,
                iou=0.45,
                verbose=False,
                device='mps'
            )[0]
            
            # Run stall detection with multiclass model
            stall_detections = self.stall_model.predict(
                source=str(tile['path']),
                classes=[3],  # stall class only
                conf=conf_threshold,
                iou=0.45,
                verbose=False,
                device='mps'
            )[0]
            
            # Extract detections
            cars = []
            stalls = []
            
            if car_detections.boxes is not None:
                cars = [box for box in car_detections.boxes]
            
            if stall_detections.boxes is not None:
                stalls = [box for box in stall_detections.boxes]
            
            print(f"Cars:{len(cars)} Stalls:{len(stalls)}")
            
            total_cars += len(cars)
            total_stalls += len(stalls)
            
            # Store results (use car_detections as primary for compatibility)
            results.append({
                'tile': tile,
                'detections': car_detections,  # for compatibility with visualization
                'cars': cars,
                'stalls': stalls,
                'objects': []  # not detected in dual-model approach
            })
        
        print(f"\n✓ Total: {total_cars} cars, {total_stalls} stalls")
        return results
    
    def stage4_stitch_and_analyze(self, tile_results: List[Dict], 
                                  num_rows: int, num_cols: int,
                                  output_dir: Path, overlap: float = 0.2) -> Dict:
        """
        Stage 4: Stitch all tiles into one coherent image and analyze occupancy.
        """
        print("\n" + "="*70)
        print("STAGE 4: Stitching & Occupancy Analysis")
        print("="*70)
        
        # Load first tile to get dimensions
        first_tile = cv2.imread(str(tile_results[0]['tile']['path']))
        tile_h, tile_w = first_tile.shape[:2]
        
        # Calculate effective step size (accounting for overlap)
        step_h = int(tile_h * (1 - overlap))
        step_w = int(tile_w * (1 - overlap))
        
        # Calculate canvas size accounting for overlap
        canvas_h = step_h * (num_rows - 1) + tile_h if num_rows > 1 else tile_h
        canvas_w = step_w * (num_cols - 1) + tile_w if num_cols > 1 else tile_w
        
        print(f"Tile size: {tile_h}x{tile_w}")
        print(f"Overlap: {overlap*100:.0f}% (step: {step_h}x{step_w})")
        print(f"Canvas: {canvas_h}x{canvas_w} for {num_rows}x{num_cols} tiles")
        
        # Create full canvas
        stitched = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Stitch tiles with proper overlap handling
        print("Stitching tiles...")
        for tr in tile_results:
            tile_img = cv2.imread(str(tr['tile']['path']))
            row = tr['tile']['row']
            col = tr['tile']['col']
            
            # Calculate position with overlap
            y1 = row * step_h
            x1 = col * step_w
            
            # Blend or average in overlap regions
            if row == 0 and col == 0:
                # First tile, just place it
                stitched[y1:y1+tile_h, x1:x1+tile_w] = tile_img
            else:
                # For overlapping tiles, average the pixels in overlap region
                y2 = min(y1 + tile_h, canvas_h)
                x2 = min(x1 + tile_w, canvas_w)
                
                # Get the region to place
                tile_region = tile_img[:y2-y1, :x2-x1]
                existing_region = stitched[y1:y2, x1:x2]
                
                # For overlapping areas (where existing is not black), average
                mask = existing_region.sum(axis=2) > 0
                blended = existing_region.copy()
                blended[mask] = (existing_region[mask].astype(np.uint16) + 
                                tile_region[mask].astype(np.uint16)) // 2
                blended[~mask] = tile_region[~mask]
                
                stitched[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        # Collect all detections in global coordinates (accounting for overlap)
        print("Collecting detections...")
        all_cars = []
        all_stalls = []
        
        for tr in tile_results:
            row = tr['tile']['row']
            col = tr['tile']['col']
            offset_y = row * step_h
            offset_x = col * step_w
            
            for box in tr['cars']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                all_cars.append([
                    int(x1 + offset_x), int(y1 + offset_y),
                    int(x2 + offset_x), int(y2 + offset_y),
                    float(conf)
                ])
            
            for box in tr['stalls']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                all_stalls.append([
                    int(x1 + offset_x), int(y1 + offset_y),
                    int(x2 + offset_x), int(y2 + offset_y),
                    float(conf)
                ])
        
        total_cars = len(all_cars)
        total_stalls = len(all_stalls)
        
        print(f"  Cars: {total_cars}")
        print(f"  Stalls: {total_stalls}")
        
        # Match cars to stalls
        print("Matching cars to stalls...")
        occupied_stalls, empty_stalls, unmatched_cars = self._match_cars_to_stalls(
            all_cars, all_stalls
        )
        
        num_occupied = len(occupied_stalls)
        num_empty = len(empty_stalls)
        occupancy_rate = (num_occupied / total_stalls * 100) if total_stalls > 0 else 0
        
        print(f"  Occupied: {num_occupied}")
        print(f"  Empty: {num_empty}")
        print(f"  Unmatched cars: {len(unmatched_cars)}")
        print(f"\n  🅿️  OCCUPANCY: {occupancy_rate:.1f}% ({num_occupied}/{total_stalls})")
        
        # Create visualization
        print("\nGenerating visualization...")
        vis = stitched.copy()
        
        # Draw vacant stalls (blue)
        for stall_idx in empty_stalls:
            x1, y1, x2, y2, _ = all_stalls[stall_idx]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)  # Blue in BGR
        
        # Draw occupied stalls (green) and cars (red)
        for match in occupied_stalls:
            stall = match['stall_box']
            car = match['car_box']
            
            cv2.rectangle(vis, (stall[0], stall[1]), (stall[2], stall[3]), (0, 255, 0), 2)  # Green in BGR
            cv2.rectangle(vis, (car[0], car[1]), (car[2], car[3]), (0, 0, 255), 2)  # Red in BGR
        
        # Draw unmatched cars (yellow)
        for car_idx in unmatched_cars:
            car = all_cars[car_idx]
            cv2.rectangle(vis, (car[0], car[1]), (car[2], car[3]), (0, 255, 255), 3)
        
        # Add text overlay
        self._add_text_overlay(vis, total_stalls, num_occupied, num_empty, occupancy_rate)
        
        # Save
        output_path = output_dir / 'overall_occupancy.jpg'
        cv2.imwrite(str(output_path), vis)
        print(f"  ✓ Saved: {output_path.name}")
        
        # Save data
        data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
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
            json.dump(data, f, indent=2)
        print(f"  ✓ Saved: {json_path.name}")
        
        return data
    
    def _match_cars_to_stalls(self, cars, stalls, iou_threshold=0.3):
        """Match cars to stalls based on IoU."""
        occupied_stalls = []
        empty_stalls = list(range(len(stalls)))
        unmatched_cars = []
        
        for car_idx, car_box in enumerate(cars):
            best_iou = 0
            best_stall_idx = -1
            
            for stall_idx in empty_stalls:
                stall_box = stalls[stall_idx]
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
                    'stall_box': stalls[best_stall_idx]
                })
                empty_stalls.remove(best_stall_idx)
            else:
                unmatched_cars.append(car_idx)
        
        return occupied_stalls, empty_stalls, unmatched_cars
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
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
    
    def _add_text_overlay(self, img, total_stalls, occupied, empty, rate):
        """Add text overlay with statistics and color-coded text."""
        h, w = img.shape[:2]
        
        # Background
        cv2.rectangle(img, (20, 20), (min(700, w-20), 250), (0, 0, 0), -1)
        cv2.rectangle(img, (20, 20), (min(700, w-20), 250), (255, 255, 255), 3)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 70
        
        cv2.putText(img, "PARKING LOT OCCUPANCY", (40, y), font, 1.2, (255, 255, 255), 3)
        y += 50
        cv2.putText(img, f"Total Stalls: {total_stalls}", (40, y), font, 1.0, (255, 255, 255), 2)
        y += 45
        
        # Color-coded text: Red for occupied (BGR: 0,0,255), Blue for vacant (BGR: 255,100,0)
        cv2.putText(img, f"Occupied: ", (40, y), font, 1.0, (255, 255, 255), 2)
        cv2.putText(img, f"{occupied}", (195, y), font, 1.0, (0, 0, 255), 2)  # Red text
        cv2.putText(img, f"  Vacant: ", (280, y), font, 1.0, (255, 255, 255), 2)
        cv2.putText(img, f"{empty}", (420, y), font, 1.0, (255, 100, 0), 2)  # Blue text
        y += 50
        
        # Occupancy rate with dynamic color
        color = (0, 255, 0) if rate < 80 else (0, 165, 255) if rate < 95 else (0, 0, 255)
        cv2.putText(img, f"Occupancy: {rate:.1f}%", (40, y), font, 1.3, color, 3)
        y += 40
        
        # Legend at bottom
        legend_y = y
        cv2.putText(img, "Legend:", (40, legend_y), font, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Green = Occupied", (140, legend_y), font, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "Blue = Vacant", (380, legend_y), font, 0.7, (255, 100, 0), 2)
    
    def run_pipeline(self, wide_area_image: Path, center_lat: float, center_lon: float,
                     zoom: int = 19, output_dir: Path = None,
                     conf_stage1: float = 0.7, conf_stage3: float = 0.25) -> Dict:
        """Run complete unified pipeline."""
        
        if output_dir is None:
            output_dir = Path("occupancy/results") / wide_area_image.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("UNIFIED PARKING OCCUPANCY DETECTION PIPELINE")
        print("="*70)
        print(f"Location: {center_lat:.6f}, {center_lon:.6f}")
        print(f"Output: {output_dir}")
        
        # Stage 1: Detect parking areas
        combined_bbox = self.stage1_detect_parking_areas(
            wide_area_image, center_lat, center_lon, zoom, conf_stage1
        )
        
        if combined_bbox is None:
            print("\n No parking areas detected")
            return None
        
        # Stage 2: Download tiles
        tiles, num_rows, num_cols = self.stage2_download_tiles(
            combined_bbox, output_dir
        )
        
        if not tiles:
            print("\n No tiles downloaded")
            return None
        
        # Stage 3: Detect objects
        tile_results = self.stage3_detect_objects(tiles, conf_stage3)
        
        # Stage 4: Stitch and analyze (0.2 overlap from stage2)
        results = self.stage4_stitch_and_analyze(
            tile_results, num_rows, num_cols, output_dir, overlap=0.2
        )
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE")
        print("="*70)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Unified parking occupancy detection")
    parser.add_argument('--image', required=True, help='Wide-area satellite image')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    parser.add_argument('--zoom', type=int, default=19, help='Zoom level (default: 19)')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--conf-stage1', type=float, default=0.7, 
                       help='Parking lot detection confidence (default: 0.7)')
    parser.add_argument('--conf-stage3', type=float, default=0.25,
                       help='Vehicle/stall detection confidence (default: 0.25)')
    
    args = parser.parse_args()
    
    pipeline = UnifiedParkingPipeline()
    
    image_path = Path(args.image)
    output_dir = Path(args.output) if args.output else None
    
    results = pipeline.run_pipeline(
        image_path, args.lat, args.lon, args.zoom,
        output_dir, args.conf_stage1, args.conf_stage3
    )
    
    if results:
        print(f"\n✓ Occupancy: {results['summary']['occupancy_rate']}%")


if __name__ == '__main__':
    main()
