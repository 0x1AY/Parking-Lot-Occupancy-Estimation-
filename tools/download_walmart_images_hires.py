#!/usr/bin/env python3
"""
Download high-resolution satellite images for Walmart parking lots using Google Maps Static API.
Similar to the training data collection approach - uses bounding boxes for large parking lots.
"""

import os
import re
import math
import csv
import requests
import urllib.parse
from pathlib import Path
from PIL import Image
from io import BytesIO
from datetime import datetime
import time

# ======================
# CONFIG
# ======================
API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')
BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"
OUTPUT_DIR = "walmart_locations/images_hires"
CSV_PATH = "walmart lots.csv"
LOG_PATH = "walmart_locations/download_log.csv"

SIZE = (640, 640)  # Match YOLO training size
SCALE = 2  # DPI scaling for higher resolution
ZOOM = 20  # High zoom for parking lot details
METERS_PER_DEG_LAT = 111320.0

# ======================
# HELPERS
# ======================
def sanitize_name(name: str) -> str:
    """Convert address to safe filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")

def meters_per_pixel(zoom, lat):
    """Calculate meters per pixel at given zoom and latitude."""
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2 ** zoom)

def fetch_static_map(lat, lon, zoom, size, scale, maptype="satellite"):
    """
    Fetch a Google Static Map image.
    
    Args:
        lat, lon: Center coordinates
        zoom: Zoom level (20 for parking lots)
        size: Tuple (width, height) in pixels
        scale: 1 or 2 for retina
        maptype: "satellite" or "roadmap"
    
    Returns:
        PIL Image object and URL used
    """
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{size[0]}x{size[1]}",
        "scale": scale,
        "maptype": maptype,
        "format": "png",
        "key": API_KEY,
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Google Maps API error {resp.status_code}: {resp.text[:200]}")
    
    return Image.open(BytesIO(resp.content)), url

def estimate_parking_lot_bounds(lat, lon, offset_meters=100):
    """
    Estimate bounding box around parking lot.
    Creates a ~200m x 200m area centered on the Walmart location.
    
    Args:
        lat, lon: Walmart center coordinates
        offset_meters: Half-width of the area in meters (default 100m = 200m total)
    
    Returns:
        (min_lat, min_lon, max_lat, max_lon)
    """
    # Convert meters to degrees
    lat_offset = offset_meters / METERS_PER_DEG_LAT
    lon_offset = offset_meters / (METERS_PER_DEG_LAT * math.cos(math.radians(lat)))
    
    min_lat = lat - lat_offset
    max_lat = lat + lat_offset
    min_lon = lon - lon_offset
    max_lon = lon + lon_offset
    
    return min_lat, min_lon, max_lat, max_lon

def download_tiled_area(site, min_lat, min_lon, max_lat, max_lon, zoom, size, scale, output_dir, log_rows):
    """
    Download multiple tiles to cover a bounding box.
    
    Returns:
        Number of tiles downloaded
    """
    avg_lat = (min_lat + max_lat) / 2
    mpp = meters_per_pixel(zoom, avg_lat)
    
    # Don't divide by scale - scale affects download resolution, not coverage
    tile_height_m = size[1] * mpp
    tile_width_m = size[0] * mpp
    
    total_height_deg = max_lat - min_lat
    total_width_deg = max_lon - min_lon
    
    total_height_m = total_height_deg * METERS_PER_DEG_LAT
    total_width_m = total_width_deg * METERS_PER_DEG_LAT * math.cos(math.radians(avg_lat))
    
    num_rows = math.ceil(total_height_m / tile_height_m)
    num_cols = math.ceil(total_width_m / tile_width_m)
    
    step_lat = total_height_deg / num_rows
    step_lon = total_width_deg / num_cols
    
    print(f"   Tiling: {num_rows} rows x {num_cols} cols = {num_rows * num_cols} tiles")
    
    tiles_downloaded = 0
    
    for row_idx in range(num_rows):
        center_lat = max_lat - (row_idx + 0.5) * step_lat
        for col_idx in range(num_cols):
            center_lon = min_lon + (col_idx + 0.5) * step_lon
            
            try:
                img, url_used = fetch_static_map(center_lat, center_lon, zoom, size, scale)
                fname = f"{site}_r{row_idx}_c{col_idx}_z{zoom}_{size[0]}x{size[1]}-{scale}x.png"
                fpath = os.path.join(output_dir, fname)
                img.save(fpath)
                print(f"   ✅ Tile [{row_idx},{col_idx}]: {fname}")
                
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "site": site,
                    "lat": center_lat,
                    "lon": center_lon,
                    "zoom": zoom,
                    "size": f"{size[0]}x{size[1]}",
                    "scale": scale,
                    "file": fpath,
                    "url": url_used,
                    "tile_row": row_idx,
                    "tile_col": col_idx,
                    "bounds": f"{min_lat},{min_lon},{max_lat},{max_lon}",
                    "error": ""
                }
                log_rows.append(log_entry)
                tiles_downloaded += 1
                
                # Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                print(f"   ❌ Error on tile [{row_idx},{col_idx}]: {e}")
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "site": site,
                    "lat": center_lat,
                    "lon": center_lon,
                    "error": str(e)
                }
                log_rows.append(log_entry)
    
    return tiles_downloaded

# ======================
# MAIN
# ======================
def main():
    print("="*70)
    print("Walmart High-Resolution Image Downloader")
    print("="*70)
    print()
    
    if not API_KEY:
        print("❌ Error: GOOGLE_MAPS_API_KEY not set")
        print("   Set it with: export GOOGLE_MAPS_API_KEY='your_api_key'")
        print("   Get a free API key at:")
        print("   https://developers.google.com/maps/documentation/maps-static/get-api-key")
        return
    
    print(f"✅ Google Maps API key found")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📐 Image size: {SIZE[0]}x{SIZE[1]} @ {SCALE}x = {SIZE[0]*SCALE}x{SIZE[1]*SCALE} pixels")
    print(f"🔍 Zoom level: {ZOOM}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read CSV file
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found")
        return
    
    locations = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            locations.append(row)
    
    print(f"📍 Found {len(locations)} Walmart locations")
    print()
    
    # Process each location
    log_rows = []
    total_tiles = 0
    
    for idx, location in enumerate(locations, 1):
        address = location['address '].strip()
        # Clean up the values - remove extra quotes and commas
        lat_str = location['lat'].strip().strip('"').rstrip(',')
        lon_str = location['long'].strip().strip('"').rstrip(',')
        lat = float(lat_str)
        lon = float(lon_str)
        
        site = sanitize_name(f"walmart_{idx:02d}_{address}")
        
        print(f"[{idx}/{len(locations)}] {address}")
        print(f"   Coordinates: ({lat:.6f}, {lon:.6f})")
        
        try:
            # Download single centered image (simpler approach)
            img, url_used = fetch_static_map(lat, lon, ZOOM, SIZE, SCALE)
            fname = f"{site}_z{ZOOM}_{SIZE[0]}x{SIZE[1]}-{SCALE}x.png"
            fpath = os.path.join(OUTPUT_DIR, fname)
            img.save(fpath)
            print(f"   ✅ Saved: {fname}")
            
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "site": site,
                "lat": lat,
                "lon": lon,
                "zoom": ZOOM,
                "size": f"{SIZE[0]}x{SIZE[1]}",
                "scale": SCALE,
                "file": fpath,
                "url": url_used,
                "error": ""
            }
            log_rows.append(log_entry)
            total_tiles += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "site": site,
                "lat": lat,
                "lon": lon,
                "error": str(e)
            }
            log_rows.append(log_entry)
        
        print()
    
    # Save log
    log_dir = Path(LOG_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(LOG_PATH, 'w', newline='') as f:
        if log_rows:
            fieldnames = log_rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)
    
    # Summary
    print("="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Total locations: {len(locations)}")
    print(f"Total images downloaded: {total_tiles}")
    print(f"📁 Images saved to: {OUTPUT_DIR}")
    print(f"📝 Log saved to: {LOG_PATH}")
    print("="*70)
    print()
    print("✅ Ready for occupancy detection!")
    print(f"   Next: python tools/detect_walmart_occupancy.py")


if __name__ == "__main__":
    main()
