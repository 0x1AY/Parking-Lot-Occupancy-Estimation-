# Parking Occupancy Detection - Streamlit Web App

🅿️ **Real-time parking occupancy analysis using satellite imagery and deep learning**

## Overview

This Streamlit web application provides an intuitive interface for the Unified Parking Occupancy Detection Pipeline. Simply enter latitude and longitude coordinates of a parking lot, and the system will:

1. Download satellite imagery from Google Maps
2. Detect parking lot boundaries
3. Identify parking stalls and vehicles
4. Calculate occupancy metrics
5. Generate visual occupancy maps

## Features

### ✨ Key Capabilities

- **Simple Input**: Just enter lat/lon coordinates
- **Real-time Processing**: Live progress tracking through all 4 pipeline stages
- **Dual-Model Architecture**: 96.3% car detection + 84% stall detection accuracy
- **Visual Results**: Color-coded occupancy maps (Green=Vacant, Red=Occupied)
- **Detailed Metrics**: Complete statistics and JSON export
- **Download Results**: Export visualizations and reports

### 🎯 Performance

- **Car Detection**: 96.3% mAP50 (high-accuracy model)
- **Stall Detection**: 84% mAP50 (multiclass model)
- **Processing Time**: ~30-60 seconds per location
- **Validated**: 10 locations, 813 stalls detected successfully

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# GPU support (optional but recommended)
# CUDA-compatible GPU with appropriate drivers
```

### Setup

1. **Clone the repository** (if not already done):

```bash
git clone https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git
cd Parking-Lot-Occupancy-Estimation-
```

2. **Install dependencies**:

```bash
# Install base requirements
pip install -r requirements.txt

# Install Streamlit
pip install -r requirements-streamlit.txt
```

3. **Verify models are present**:

```bash
# Localization model
ls datasets/apklot/apklot_stage1/weights/best.pt

# Car detection model
ls parking_runs/yolo11m_parking_augmented2/weights/best.pt

# Stall detection model
ls parking_runs/yolo11m_multilabel/weights/best.pt
```

## Usage

### Starting the App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Initialize Pipeline** (First time):

   - Configure model paths in the sidebar (defaults are pre-filled)
   - Enter your Google Maps API key
   - Click "🔄 Initialize Pipeline"
   - Wait for models to load (~10-20 seconds)

2. **Analyze a Location**:

   - Enter latitude and longitude coordinates
   - (Optional) Provide a location name
   - Click "🚀 Analyze Parking Occupancy"
   - Watch the progress through 4 stages:
     - Stage 1: Detecting parking lot areas
     - Stage 2: Downloading high-resolution tiles
     - Stage 3: Detecting cars and parking stalls
     - Stage 4: Processing complete

3. **View Results**:
   - **Metrics**: Total stalls, occupancy rate, processing time
   - **Visualization**: Color-coded occupancy map
   - **Details**: Detection statistics and location info
   - **Export**: Download visualization and JSON report

### Configuration Options

#### Sidebar Settings

- **Model Paths**: Customize paths to detection models
- **Localization Zoom**: Adjust initial detection zoom (17-20)
- **Tile Zoom**: High-resolution tile zoom level (19-21)
- **Confidence Threshold**: Minimum detection confidence (0.1-0.9)
- **IoU Threshold**: Car-to-stall matching threshold (0.1-0.9)

#### Example Locations

The app includes pre-configured coordinates for sample locations:

| Location              | Latitude  | Longitude  |
| --------------------- | --------- | ---------- |
| Walmart Gerrard St    | 43.668734 | -79.340158 |
| Walmart Dufferin St   | 43.666156 | -79.444583 |
| Walmart St Clair Ave  | 43.675844 | -79.505278 |
| Walmart Islington Ave | 43.665417 | -79.583611 |
| Walmart Lawrence Ave  | 43.712778 | -79.473333 |

## How It Works

### Pipeline Architecture

The app uses a 4-stage unified pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Parking Lot Localization (APKLOT Model)           │
│ • Downloads wide-area satellite image (zoom 19)             │
│ • Detects parking lot boundaries                            │
│ • Calculates geographic bounds                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: High-Resolution Tile Download                      │
│ • Calculates tile grid with 20% overlap                     │
│ • Downloads tiles at zoom 20 (higher detail)                │
│ • Covers entire parking area                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Dual-Model Object Detection                        │
│ • Car Model: 96.3% mAP50 (specialized car detection)        │
│ • Stall Model: 84% mAP50 (stalls, boundaries, objects)      │
│ • Processes all tiles with both models                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Stitching & Occupancy Calculation                  │
│ • Stitches tiles into coherent image                        │
│ • IoU-based car-to-stall matching                           │
│ • Calculates occupancy metrics                              │
│ • Generates visualization and reports                       │
└─────────────────────────────────────────────────────────────┘
```

### Dual-Model Strategy

**Why Two Models?**

- **Specialized Performance**: Each model optimized for specific task
- **Higher Accuracy**: +14.6% improvement in car detection (96.3% vs 84%)
- **Robust Detection**: Car model focuses on vehicles, stall model on infrastructure

## Output Files

Results are saved in temporary directories during processing:

```
results/
└── <location_name>/
    ├── <location_name>_z19.png        # Wide area image
    ├── tiles/                          # High-res tile grid
    │   ├── tile_r0_c0.png
    │   ├── tile_r0_c1.png
    │   └── ...
    ├── overall_occupancy.jpg          # Final visualization
    └── overall_occupancy.json         # Metrics JSON
```

### Visualization Legend

- 🟢 **Green boxes**: Vacant parking stalls
- 🔴 **Red boxes**: Occupied stalls (with cars detected)
- 🟡 **Yellow boxes**: Unmatched cars (not in stalls)
- 🔵 **Blue boxes**: Empty stalls

## Troubleshooting

### Common Issues

**Models not loading:**

```bash
# Verify model paths exist
ls -la parking_runs/yolo11m_parking_augmented2/weights/best.pt
ls -la parking_runs/yolo11m_multilabel/weights/best.pt
ls -la datasets/apklot/apklot_stage1/weights/best.pt
```

**Google Maps API errors:**

- Verify API key is valid
- Check API quota limits
- Ensure Static Maps API is enabled

**Out of memory:**

- Reduce batch size in detection models
- Use CPU instead of GPU (slower but less memory)
- Close other applications

**Slow processing:**

- Use GPU acceleration (CUDA/MPS)
- Reduce tile resolution
- Lower detection confidence thresholds

## Technical Specifications

### System Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (optional)
- **Storage**: 10GB free space
- **Internet**: Stable connection for API calls

### Model Specifications

| Model          | Type     | mAP50 | Size  | Purpose                         |
| -------------- | -------- | ----- | ----- | ------------------------------- |
| APKLOT Stage 1 | YOLOv11m | 83.5% | ~50MB | Parking lot localization        |
| Car Detection  | YOLOv11m | 96.3% | ~50MB | High-accuracy vehicle detection |
| Multiclass     | YOLOv11m | 84.0% | ~50MB | Stall and object detection      |

### Performance Metrics

- **Processing Time**: 30-60 seconds per location
- **Accuracy**: 96.3% car detection, 84% stall detection
- **Scalability**: Handles parking lots of any size
- **Success Rate**: 100% on validated locations

## API Reference

### UnifiedParkingPipeline.process_location()

```python
pipeline.process_location(
    location_name: str,         # Name for this location
    lat: float,                 # Latitude coordinate
    lon: float,                 # Longitude coordinate
    output_dir: Path,           # Output directory
    localization_zoom: int = 19,  # Initial zoom level
    tile_zoom: int = 20,        # Tile zoom level
    conf_threshold: float = 0.25, # Detection confidence
    iou_threshold: float = 0.3   # IoU matching threshold
) -> Dict
```

**Returns:**

```python
{
    'location_name': str,
    'latitude': float,
    'longitude': float,
    'total_stalls': int,
    'occupied_stalls': int,
    'empty_stalls': int,
    'occupancy_rate': float,
    'cars_detected': int,
    'unmatched_cars': int,
    'result_path': str,
    'timestamp': str,
    'processing_success': bool
}
```

## Development

### Running in Development Mode

```bash
# Enable debug mode
streamlit run app.py --server.runOnSave true

# Custom port
streamlit run app.py --server.port 8502

# Disable CORS
streamlit run app.py --server.enableCORS false
```

### Extending the App

The app is modular and can be extended with:

- Historical occupancy tracking
- Multiple location comparison
- Real-time camera integration
- Occupancy prediction models
- Mobile-responsive layouts

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{parking-occupancy-2025,
  author = {Yiwere, Aminu and Olagundoye, Olatunji},
  title = {Parking Lot Occupancy Estimation Using Deep Learning},
  year = {2025},
  institution = {Northeastern University},
  url = {https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-}
}
```

## License

MIT License - See [LICENSE](LICENSE) file for details

## Support

For issues, questions, or contributions:

- **GitHub Issues**: [Report bugs or request features](https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-/issues)
- **Email**: Contact project maintainers
- **Documentation**: See main [README.md](README.md) for detailed project information

## Acknowledgments

- **YOLOv11**: Ultralytics team for the detection framework
- **APKLOT Dataset**: For parking lot localization training data
- **Google Maps API**: For satellite imagery access
- **Streamlit**: For the web app framework
- **Northeastern University**: Deep Learning course support

---

**Built with ❤️ for smart city applications**
