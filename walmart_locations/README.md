# Walmart Parking Lot Occupancy Analysis

## Challenge

The downloaded Bing Maps satellite images are single 256x256 tiles, which are:

- Too small for detailed parking lot analysis
- Different resolution than training data (640x640)
- May not show parking stall markings clearly

## Solutions

### Option 1: Use Google Maps Static API (Recommended)

1. Get a free Google Maps API key from: https://developers.google.com/maps/documentation/maps-static/get-api-key
2. Set the environment variable:
   ```bash
   export GOOGLE_MAPS_API_KEY='your_api_key_here'
   ```
3. Re-run the download script:
   ```bash
   python tools/download_walmart_images.py
   ```

This will download high-resolution 640x640 images that match the training data format.

### Option 2: Manual Approach

Since stalls aren't being detected automatically:

1. **Define stalls manually** for each Walmart location:

   - Create a JSON file with stall coordinates for each location
   - One-time manual annotation using a tool like LabelImg
   - Save as `walmart_locations/stall_configs/location_name.json`

2. **Run car-only detection**:
   - Use the car detection model (class 0 only)
   - Match detected cars to manually defined stalls
   - Calculate occupancy from this

### Option 3: Use the Original Model (Cars Only)

Since we can detect cars but not stalls in these images, we can:

1. Use the single-class car detection model
2. Create a simple grid-based approach:
   - Manually define parking area boundaries
   - Create a regular grid of assumed stall locations
   - Match detected cars to grid cells
   - Estimate occupancy

## Current Status

✅ Downloaded 10 Walmart location images from Bing Maps
✅ Created occupancy detection pipeline
⚠️ Model doesn't detect stalls in these images (different imagery source)
❌ Need either better images or manual stall definitions

## Recommended Next Steps

1. **Get Google Maps API key** for better quality images (FREE for development)
2. Or **manually annotate stalls** for a few test locations
3. Or **use car-only detection** with estimated stall grid

## Files Created

- `tools/download_walmart_images.py` - Download satellite images
- `tools/detect_walmart_occupancy.py` - Run occupancy detection
- `walmart lots.csv` - List of 10 Toronto Walmart locations
- `walmart_locations/images/` - Downloaded satellite images (Bing Maps tiles)
- `walmart_locations/results/` - Occupancy detection results (0 stalls detected)
