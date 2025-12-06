# Streamlit App User Guide

## Application Interface Overview

### Main Screen Layout

The Streamlit app is organized into several key sections:

#### 1. Header Section

- **Title**: "🅿️ Parking Occupancy Detection System"
- **Subtitle**: "Real-time parking occupancy analysis using satellite imagery and deep learning"
- **Status Indicator**: Shows if pipeline is initialized

#### 2. Sidebar Configuration Panel

**Model Settings:**

- Localization Model path input
- Car Detection Model path input
- Stall Detection Model path input
- Google Maps API Key (password field)

**Detection Parameters:**

- Localization Zoom slider (17-20, default: 19)
- Tile Zoom slider (19-21, default: 20)
- Confidence Threshold slider (0.1-0.9, default: 0.25)
- IoU Threshold slider (0.1-0.9, default: 0.3)

**Initialize Button:**

- Large blue button to load all models
- Shows loading spinner during initialization

#### 3. Main Content Area

**Before Initialization:**

- System overview with 3 columns:
  - Dual-Model Architecture specs
  - 4-Stage Pipeline description
  - Proven Results statistics
- Example locations table with 5 sample coordinates

**After Initialization:**

- Success message (green box)
- Location input form:
  - Latitude input (number field)
  - Longitude input (number field)
  - Location Name input (optional text field)
- "🚀 Analyze Parking Occupancy" button

**During Processing:**

- Progress bar (0-100%)
- Status text showing current stage:
  - "🔍 Stage 1/4: Detecting parking lot areas..."
  - "📥 Stage 2/4: Downloading high-resolution tiles..."
  - "🚗 Stage 3/4: Detecting cars and parking stalls..."
  - "✅ Stage 4/4: Processing complete!"

**Results Display:**

1. **Metrics Row** (4 columns):

   - Total Stalls (number)
   - Occupied Stalls (number)
   - Occupancy Rate (percentage)
   - Processing Time (seconds)

2. **Occupancy Visualization**:

   - Full-width satellite image overlay
   - Color-coded bounding boxes:
     - Green: Vacant stalls
     - Red: Occupied stalls (with cars)
     - Yellow: Unmatched cars
   - Caption explaining legend

3. **Detailed Metrics** (2 columns):

   - Left: Detection Statistics
     - Cars Detected
     - Stalls Detected
     - Occupied Stalls
     - Vacant Stalls
   - Right: Location Details
     - Latitude
     - Longitude
     - Location Name
     - Analysis Date

4. **Raw JSON Output**:

   - Expandable section
   - Complete JSON response

5. **Download Buttons** (2 columns):
   - Download Visualization (JPG)
   - Download JSON Report

#### 4. Footer

- System information
- Model performance specs
- Copyright notice

## User Workflow

### Step-by-Step Guide

1. **Start the App**

   ```bash
   ./run_app.sh
   ```

   Browser opens to http://localhost:8501

2. **Initialize Pipeline**

   - Review model paths in sidebar (or use defaults)
   - Enter Google Maps API key
   - Click "🔄 Initialize Pipeline"
   - Wait 10-20 seconds for models to load
   - Look for green success message

3. **Enter Location**

   - Type latitude (e.g., 43.668734)
   - Type longitude (e.g., -79.340158)
   - Optionally name the location
   - Click "🚀 Analyze Parking Occupancy"

4. **Monitor Progress**

   - Watch progress bar advance through 4 stages
   - Read status messages
   - Wait 30-60 seconds for processing

5. **Review Results**

   - Check occupancy metrics at top
   - Examine visualization
   - Review detailed statistics
   - Expand JSON output if needed

6. **Export Results**

   - Click "⬇️ Download Visualization" for image
   - Click "⬇️ Download JSON Report" for data
   - Files save to browser's download folder

7. **Analyze Another Location**
   - Enter new coordinates
   - Click analyze button again
   - Previous results replaced with new ones

## Tips and Tricks

### Finding Coordinates

1. **Google Maps:**

   - Right-click on parking lot center
   - Select "What's here?"
   - Copy lat/lon from bottom info card

2. **Example Locations in App:**
   - Pre-configured Walmart stores
   - Copy coordinates directly

### Optimal Settings

**For Best Results:**

- Localization Zoom: 19 (good balance)
- Tile Zoom: 20 (high detail without too many tiles)
- Confidence Threshold: 0.25 (standard YOLO default)
- IoU Threshold: 0.3 (works well for overhead parking views)

**For Faster Processing:**

- Localization Zoom: 18 (covers more area)
- Tile Zoom: 19 (fewer tiles to download)
- Confidence Threshold: 0.3 (fewer detections)

**For Higher Accuracy:**

- Localization Zoom: 19 (precise parking area)
- Tile Zoom: 21 (maximum detail)
- Confidence Threshold: 0.2 (catch more detections)
- IoU Threshold: 0.25 (stricter matching)

### Troubleshooting

**"Failed to initialize pipeline":**

- Check model paths exist
- Verify sufficient RAM/VRAM
- Try restarting app

**"No parking areas detected":**

- Adjust localization zoom
- Lower confidence threshold
- Verify coordinates are correct
- Try slightly different lat/lon

**Slow processing:**

- Use GPU if available
- Reduce tile zoom level
- Close other applications
- Check internet connection speed

**API errors:**

- Verify API key is valid
- Check API quota limits
- Ensure Static Maps API is enabled
- Try again after a few minutes

## Color Legend

### Visualization Colors

- **🟢 Green Boxes**: Vacant parking stalls

  - No car detected in stall area
  - Available for parking

- **🔴 Red Boxes**: Occupied stalls

  - Car detected with IoU > threshold
  - Space is in use

- **🟡 Yellow Boxes**: Unmatched cars

  - Vehicle detected but not in stall
  - Could be driving through or improperly parked

- **🔵 Blue Boxes**: Empty stalls (alternative view)
  - Same as green, different color scheme option

## Performance Expectations

### Processing Times

| Parking Lot Size  | Tiles | Typical Time  |
| ----------------- | ----- | ------------- |
| Small (1-2 rows)  | 4-6   | 20-30 seconds |
| Medium (2-3 rows) | 6-9   | 30-45 seconds |
| Large (3+ rows)   | 9-12  | 45-60 seconds |
| Very Large        | 12+   | 60-90 seconds |

### Accuracy

- **Car Detection**: 96.3% mAP50
- **Stall Detection**: 84% mAP50
- **Occupancy Estimation**: Typically accurate within 2-3 stalls
- **False Positives**: <5% (mostly shadows or objects)
- **False Negatives**: <5% (mainly heavily occluded vehicles)

## Data Privacy

- Satellite images downloaded temporarily
- Results stored only during session
- No data persisted after app closes
- Uses public Google Maps imagery
- No personal information collected

## System Requirements

### Minimum

- 8GB RAM
- Dual-core CPU
- 10GB disk space
- Internet connection

### Recommended

- 16GB RAM
- Quad-core CPU
- NVIDIA GPU with 4GB+ VRAM
- Fast internet (10+ Mbps)
- SSD storage

## Support

For issues or questions:

- Check STREAMLIT_README.md for detailed docs
- Review main README.md for project info
- Submit GitHub issues
- Contact project maintainers
