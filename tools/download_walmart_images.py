#!/usr/bin/env python3
"""
Download satellite images for Walmart parking lots from Google Maps Static API.
Uses the coordinates from walmart lots.csv to fetch high-resolution satellite imagery.
"""

import os
import csv
import requests
from pathlib import Path
import time

# Configuration
WALMART_CSV = "walmart lots.csv"
OUTPUT_DIR = "walmart_locations/images"
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')

# Image parameters (matching training data resolution)
IMAGE_SIZE = "640x640"  # Match YOLO training size
ZOOM_LEVEL = 20  # High zoom for parking lot details
MAP_TYPE = "satellite"

def sanitize_filename(address):
    """Convert address to safe filename."""
    # Remove special characters, keep alphanumeric and spaces
    safe = ''.join(c if c.isalnum() or c in (' ', '_') else '_' for c in address)
    # Replace multiple spaces/underscores with single underscore
    safe = '_'.join(safe.split())
    return safe[:100]  # Limit length


def download_satellite_image(lat, lon, address, output_path):
    """
    Download satellite image from Google Maps Static API.
    
    Args:
        lat: Latitude
        lon: Longitude
        address: Location address (for filename)
        output_path: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    if not GOOGLE_MAPS_API_KEY:
        print("⚠️  Warning: GOOGLE_MAPS_API_KEY not set")
        print("   Set it with: export GOOGLE_MAPS_API_KEY='your_api_key'")
        print("   Or get a free alternative API key")
        return False
    
    # Google Maps Static API URL
    url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params = {
        'center': f"{lat},{lon}",
        'zoom': ZOOM_LEVEL,
        'size': IMAGE_SIZE,
        'maptype': MAP_TYPE,
        'key': GOOGLE_MAPS_API_KEY,
        'scale': 2  # Higher resolution
    }
    
    try:
        print(f"   Downloading: {address[:50]}...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✅ Saved: {output_path.name}")
            return True
        else:
            print(f"   ❌ Error {response.status_code}: {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_bing_satellite_image(lat, lon, address, output_path):
    """
    Alternative: Download from Bing Maps (no API key required for static tiles).
    
    Uses Bing Maps Tile System for satellite imagery.
    """
    import math
    
    def lat_lon_to_tile(lat, lon, zoom):
        """Convert lat/lon to tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x_tile, y_tile
    
    x_tile, y_tile = lat_lon_to_tile(lat, lon, ZOOM_LEVEL)
    
    # Bing Maps tile URL (aerial imagery)
    # Format: http://ecn.t{0-3}.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=0
    # Need to convert tile to quadkey
    quadkey = ''
    for i in range(ZOOM_LEVEL, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x_tile & mask) != 0:
            digit += 1
        if (y_tile & mask) != 0:
            digit += 2
        quadkey += str(digit)
    
    url = f"http://ecn.t0.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=0"
    
    try:
        print(f"   Downloading from Bing: {address[:50]}...")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✅ Saved: {output_path.name}")
            return True
        else:
            print(f"   ❌ Error {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    print("="*70)
    print("Walmart Parking Lot Image Downloader")
    print("="*70)
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Read CSV file
    if not os.path.exists(WALMART_CSV):
        print(f"❌ Error: {WALMART_CSV} not found")
        return
    
    locations = []
    with open(WALMART_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            locations.append(row)
    
    print(f"📍 Found {len(locations)} Walmart locations")
    print()
    
    # Download images
    successful = 0
    failed = 0
    
    use_bing = not GOOGLE_MAPS_API_KEY  # Use Bing if no Google API key
    
    if use_bing:
        print("ℹ️  Using Bing Maps (no API key required)")
        print()
    
    for idx, location in enumerate(locations, 1):
        address = location['address '].strip()  # Note: CSV has space in header
        lat = float(location['lat'].strip())
        lon = float(location['long'].strip().rstrip(','))  # Remove trailing comma
        
        # Create filename
        safe_name = sanitize_filename(address)
        filename = f"walmart_{idx:02d}_{safe_name}.jpg"
        output_path = output_dir / filename
        
        print(f"[{idx}/{len(locations)}] {address}")
        print(f"   Coordinates: ({lat}, {lon})")
        
        # Download image
        if use_bing:
            success = download_bing_satellite_image(lat, lon, address, output_path)
        else:
            success = download_satellite_image(lat, lon, address, output_path)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Rate limiting
        if idx < len(locations):
            time.sleep(0.5)  # Be nice to the API
        
        print()
    
    # Summary
    print("="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Total locations: {len(locations)}")
    print(f"✅ Successful:   {successful}")
    print(f"❌ Failed:       {failed}")
    print(f"📁 Images saved to: {output_dir}")
    print("="*70)
    
    if use_bing:
        print()
        print("ℹ️  Note: Bing Maps tiles may have different resolution than training data.")
        print("   For best results, set GOOGLE_MAPS_API_KEY environment variable.")
        print("   Get free API key at: https://developers.google.com/maps/documentation/maps-static/get-api-key")


if __name__ == "__main__":
    main()
