#!/usr/bin/env python3
"""
Tile Coverage Planning for Multi-Stage Parking Detection
=========================================================
Converts detected parking lot polygons from Stage 1 model into high-resolution
tile grids for Stage 2 vehicle detection.

Process:
1. Load parking lot detection results (segmentation masks)
2. Extract polygon boundaries and calculate geographic bounding boxes
3. Generate optimal tile grid with overlap for seamless stitching
4. Output tile specifications for high-resolution download

Usage:
    python tools/plan_tile_coverage.py --image path/to/image.png --model datasets/apklot/apklot_stage1/weights/best.pt
    python tools/plan_tile_coverage.py --dir walmart_locations/wide_area_z19/
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MPLPolygon


class TilePlanner:
    """Plan tile coverage for high-resolution parking lot scanning."""
    
    def __init__(self, image_path: Path, center_lat: float, center_lon: float, zoom: int):
        """
        Initialize tile planner.
        
        Args:
            image_path: Path to the wide-area satellite image
            center_lat: Latitude of image center
            center_lon: Longitude of image center
            zoom: Zoom level of the input image
        """
        self.image_path = image_path
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        
        # Load image to get dimensions
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        self.img_height, self.img_width = self.image.shape[:2]
        
        # Calculate meters per pixel at this zoom level and latitude
        # Reference: https://wiki.openstreetmap.org/wiki/Zoom_levels
        self.meters_per_pixel = self._calculate_meters_per_pixel()
        
    def _calculate_meters_per_pixel(self) -> float:
        """
        Calculate meters per pixel at the given zoom level and latitude.
        
        Formula: meters_per_pixel = (Earth_circumference * cos(lat)) / (2^(zoom+8))
        """
        import math
        earth_circumference = 40075016.686  # meters at equator
        lat_rad = math.radians(self.center_lat)
        
        # Meters per pixel at equator for this zoom
        mpp_equator = earth_circumference / (2 ** (self.zoom + 8))
        
        # Adjust for latitude
        mpp = mpp_equator * math.cos(lat_rad)
        
        return mpp
    
    def pixel_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to lat/lon.
        
        Args:
            x: Pixel x coordinate (0 = left edge)
            y: Pixel y coordinate (0 = top edge)
        
        Returns:
            (latitude, longitude)
        """
        import math
        
        # Calculate offset from center in pixels
        dx_pixels = x - (self.img_width / 2)
        dy_pixels = (self.img_height / 2) - y  # Invert Y axis
        
        # Convert to meters
        dx_meters = dx_pixels * self.meters_per_pixel
        dy_meters = dy_pixels * self.meters_per_pixel
        
        # Convert meters to degrees
        # 1 degree latitude ≈ 111,319.9 meters
        # 1 degree longitude varies with latitude
        lat_offset = dy_meters / 111319.9
        lon_offset = dx_meters / (111319.9 * math.cos(math.radians(self.center_lat)))
        
        lat = self.center_lat + lat_offset
        lon = self.center_lon + lon_offset
        
        return lat, lon
    
    def extract_polygon_from_mask(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        """
        Extract polygon boundary from segmentation mask.
        
        Args:
            mask: Binary mask (H x W)
        
        Returns:
            List of (x, y) pixel coordinates forming the polygon
        """
        # Find contours
        contours, _ = cv2.findContours(
            (mask > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify polygon (Douglas-Peucker)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to list of tuples
        polygon = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
        
        return polygon
    
    def calculate_bounding_box(self, polygon: List[Tuple[float, float]]) -> Dict:
        """
        Calculate geographic bounding box from polygon.
        
        Args:
            polygon: List of (x, y) pixel coordinates
        
        Returns:
            Dict with min_lat, max_lat, min_lon, max_lon in degrees
        """
        if not polygon:
            return None
        
        # Convert all points to lat/lon
        latlon_points = [self.pixel_to_latlon(x, y) for x, y in polygon]
        
        lats = [lat for lat, lon in latlon_points]
        lons = [lon for lat, lon in latlon_points]
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
            'center_lat': (min(lats) + max(lats)) / 2,
            'center_lon': (min(lons) + max(lons)) / 2,
            'polygon_pixels': polygon,
            'polygon_latlon': latlon_points
        }
    
    def plan_tile_grid(self, bbox: Dict, target_zoom: int = 20, 
                       tile_size: int = 640, overlap: float = 0.2) -> List[Dict]:
        """
        Generate tile grid for high-resolution scanning.
        
        Args:
            bbox: Bounding box dict from calculate_bounding_box()
            target_zoom: Zoom level for tiles (20 = 0.3m/pixel)
            tile_size: Tile size in pixels (640x640 for YOLO)
            overlap: Overlap fraction (0.2 = 20% overlap for stitching)
        
        Returns:
            List of tile specifications with center lat/lon and size
        """
        import math
        
        # Calculate meters per pixel at target zoom
        earth_circumference = 40075016.686
        lat_rad = math.radians(bbox['center_lat'])
        target_mpp = (earth_circumference * math.cos(lat_rad)) / (2 ** (target_zoom + 8))
        
        # Calculate tile coverage in degrees
        tile_meters = tile_size * target_mpp
        tile_lat_deg = tile_meters / 111319.9
        tile_lon_deg = tile_meters / (111319.9 * math.cos(lat_rad))
        
        # Calculate effective step (accounting for overlap)
        step_factor = 1 - overlap
        step_lat = tile_lat_deg * step_factor
        step_lon = tile_lon_deg * step_factor
        
        # Calculate number of tiles needed
        lat_span = bbox['max_lat'] - bbox['min_lat']
        lon_span = bbox['max_lon'] - bbox['min_lon']
        
        num_tiles_lat = int(np.ceil(lat_span / step_lat))
        num_tiles_lon = int(np.ceil(lon_span / step_lon))
        
        # Ensure at least 1 tile in each direction
        num_tiles_lat = max(1, num_tiles_lat)
        num_tiles_lon = max(1, num_tiles_lon)
        
        # Generate tile centers
        tiles = []
        for i in range(num_tiles_lat):
            for j in range(num_tiles_lon):
                # Calculate tile center
                center_lat = bbox['min_lat'] + (i + 0.5) * step_lat
                center_lon = bbox['min_lon'] + (j + 0.5) * step_lon
                
                # Clamp to bounding box (with margin)
                center_lat = max(bbox['min_lat'], min(bbox['max_lat'], center_lat))
                center_lon = max(bbox['min_lon'], min(bbox['max_lon'], center_lon))
                
                tiles.append({
                    'tile_id': f"{i}_{j}",
                    'lat': center_lat,
                    'lon': center_lon,
                    'zoom': target_zoom,
                    'size': tile_size,
                    'scale': 2,  # Retina quality
                    'row': i,
                    'col': j,
                    'overlap': overlap
                })
        
        return tiles


def process_image(image_path: Path, model: YOLO, center_lat: float, center_lon: float,
                  zoom: int, conf_threshold: float = 0.5, visualize: bool = True) -> Dict:
    """
    Process a wide-area image and plan tile coverage for all detected parking lots.
    
    Args:
        image_path: Path to input image
        model: Loaded YOLO model
        center_lat: Latitude of image center
        center_lon: Longitude of image center
        zoom: Zoom level of input image
        conf_threshold: Confidence threshold for detections
        visualize: Whether to create visualization
    
    Returns:
        Dict with detection results and tile plans
    """
    print(f"\n{'='*70}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*70}")
    print(f"Center: {center_lat:.6f}, {center_lon:.6f}")
    print(f"Zoom: {zoom}")
    
    # Initialize tile planner
    planner = TilePlanner(image_path, center_lat, center_lon, zoom)
    print(f"Image size: {planner.img_width}x{planner.img_height} pixels")
    print(f"Resolution: {planner.meters_per_pixel:.2f} meters/pixel")
    
    # Run detection
    print(f"\nRunning parking lot detection (conf={conf_threshold})...")
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        iou=0.45,
        verbose=False,
        device='mps'
    )[0]
    
    num_detections = len(results.boxes) if results.boxes is not None else 0
    print(f"✓ Detected {num_detections} parking lot(s)")
    
    if num_detections == 0:
        return {'image': image_path.name, 'parking_lots': []}
    
    # Process each parking lot
    parking_lots = []
    total_tiles = 0
    
    for idx, (box, mask) in enumerate(zip(results.boxes, results.masks)):
        conf = box.conf.item()
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        print(f"\nParking Lot #{idx + 1} (confidence: {conf:.3f})")
        
        # Extract polygon from mask
        mask_data = mask.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask_data, (planner.img_width, planner.img_height))
        polygon = planner.extract_polygon_from_mask(mask_resized)
        
        if not polygon:
            print(f"  ⚠️ Failed to extract polygon, skipping")
            continue
        
        # Calculate bounding box
        bbox = planner.calculate_bounding_box(polygon)
        
        if bbox is None:
            print(f"  ⚠️ Failed to calculate bounding box, skipping")
            continue
        
        lat_span = bbox['max_lat'] - bbox['min_lat']
        lon_span = bbox['max_lon'] - bbox['min_lon']
        
        print(f"  Geographic bounds:")
        print(f"    Latitude:  {bbox['min_lat']:.6f} to {bbox['max_lat']:.6f} (span: {lat_span*111319.9:.1f}m)")
        print(f"    Longitude: {bbox['min_lon']:.6f} to {bbox['max_lon']:.6f} (span: {lon_span*111319.9*np.cos(np.radians(center_lat)):.1f}m)")
        
        # Plan tile grid
        tiles = planner.plan_tile_grid(bbox, target_zoom=20, tile_size=640, overlap=0.2)
        
        print(f"  Tile grid: {len(tiles)} tiles")
        
        # Calculate grid dimensions
        if tiles:
            rows = max(t['row'] for t in tiles) + 1
            cols = max(t['col'] for t in tiles) + 1
            print(f"  Grid layout: {rows} rows × {cols} cols")
        
        parking_lots.append({
            'id': idx + 1,
            'confidence': conf,
            'bbox_pixels': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
            'bbox_geo': bbox,
            'tiles': tiles
        })
        
        total_tiles += len(tiles)
    
    print(f"\n{'='*70}")
    print(f"Summary: {num_detections} parking lots, {total_tiles} total tiles")
    print(f"{'='*70}")
    
    result = {
        'image': image_path.name,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'zoom': zoom,
        'parking_lots': parking_lots,
        'total_tiles': total_tiles
    }
    
    # Visualize
    if visualize and parking_lots:
        visualize_tile_plan(planner, parking_lots, results)
    
    return result


def visualize_tile_plan(planner: TilePlanner, parking_lots: List[Dict], 
                        detection_results, save_dir: Path = None):
    """Visualize parking lot detections and tile grid overlay."""
    
    image_rgb = cv2.cvtColor(planner.image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.imshow(image_rgb)
    ax.set_xlim(0, planner.img_width)
    ax.set_ylim(planner.img_height, 0)
    ax.axis('off')
    
    # Color map for different parking lots
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, lot in enumerate(parking_lots):
        color = colors[idx % 10]
        
        # Draw polygon boundary
        polygon_pixels = lot['bbox_geo']['polygon_pixels']
        if polygon_pixels:
            poly = MPLPolygon(
                polygon_pixels, closed=True,
                linewidth=3, edgecolor=color,
                facecolor='none', linestyle='-'
            )
            ax.add_patch(poly)
        
        # Draw bounding box
        bbox = lot['bbox_pixels']
        rect = Rectangle(
            (bbox['x1'], bbox['y1']),
            bbox['x2'] - bbox['x1'],
            bbox['y2'] - bbox['y1'],
            linewidth=2, edgecolor=color, facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        
        # Label
        label = f"Lot #{lot['id']} ({lot['confidence']:.2f})\n{len(lot['tiles'])} tiles"
        ax.text(
            bbox['x1'], bbox['y1'] - 20, label,
            fontsize=10, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8)
        )
        
        # Draw tile grid (simplified visualization)
        # Show tile centers as small dots
        for tile in lot['tiles'][:50]:  # Limit to first 50 for visibility
            # Convert tile center back to pixels
            x_pixel = (tile['lon'] - planner.center_lon) * (111319.9 * np.cos(np.radians(planner.center_lat))) / planner.meters_per_pixel + planner.img_width / 2
            y_pixel = planner.img_height / 2 - (tile['lat'] - planner.center_lat) * 111319.9 / planner.meters_per_pixel
            
            ax.plot(x_pixel, y_pixel, 'o', color=color, markersize=3, alpha=0.6)
    
    plt.title(f"Parking Lot Detection & Tile Planning: {planner.image_path.name}", 
              fontsize=16, weight='bold')
    plt.tight_layout()
    
    # Save
    if save_dir is None:
        save_dir = planner.image_path.parent / 'tile_plans'
    save_dir.mkdir(exist_ok=True)
    
    output_path = save_dir / f"{planner.image_path.stem}_tile_plan.jpg"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plan tile coverage for detected parking lots"
    )
    parser.add_argument(
        '--image', type=str,
        help='Path to wide-area satellite image'
    )
    parser.add_argument(
        '--lat', type=float,
        help='Latitude of image center'
    )
    parser.add_argument(
        '--lon', type=float,
        help='Longitude of image center'
    )
    parser.add_argument(
        '--zoom', type=int, default=19,
        help='Zoom level of input image (default: 19)'
    )
    parser.add_argument(
        '--model', type=str,
        default='datasets/apklot/apklot_stage1/weights/best.pt',
        help='Path to parking lot detection model'
    )
    parser.add_argument(
        '--conf', type=float, default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output JSON file for tile specifications'
    )
    
    args = parser.parse_args()
    
    if not args.image or not args.lat or not args.lon:
        print("❌ Required arguments: --image, --lat, --lon")
        parser.print_help()
        return
    
    # Load model
    print("=" * 70)
    print("Tile Coverage Planning - Multi-Stage Parking Detection")
    print("=" * 70)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\nLoading model: {model_path}")
    model = YOLO(str(model_path))
    print("✓ Model loaded")
    
    # Process image
    image_path = Path(args.image)
    result = process_image(
        image_path, model, args.lat, args.lon, args.zoom,
        conf_threshold=args.conf, visualize=True
    )
    
    # Save tile specifications
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / 'tile_plans' / f"{image_path.stem}_tiles.json"
    
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Tile specifications saved: {output_path}")
    print("\nNext steps:")
    print(f"  1. Download {result['total_tiles']} high-resolution tiles (zoom 20)")
    print(f"  2. Run vehicle detection on each tile")
    print(f"  3. Stitch results and calculate occupancy")


if __name__ == '__main__':
    main()
