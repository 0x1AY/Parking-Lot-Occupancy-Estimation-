# Parking Lot Occupancy Web Application

This folder contains the Streamlit web application for real-time parking lot occupancy detection.

## Quick Start

```bash
# Install dependencies
pip install -r requirements-streamlit.txt

# Run the app
./run_app.sh
```

Or manually:

```bash
streamlit run app.py
```

## Files in This Directory

- **app.py** - Main Streamlit application (459 lines)
- **STREAMLIT_README.md** - Complete technical documentation
- **STREAMLIT_USER_GUIDE.md** - User interface guide
- **run_app.sh** - Quick start script
- **test_streamlit_pipeline.py** - Testing script
- **requirements-streamlit.txt** - Streamlit dependency

## Features

- 🗺️ **Coordinate Input** - Enter lat/lon for any parking lot
- 🔄 **Real-time Processing** - 4-stage pipeline with progress tracking
- 📊 **Metrics Dashboard** - Total stalls, occupied, occupancy rate
- 🎨 **Visual Results** - Color-coded occupancy map (Green=Vacant, Red=Occupied)
- ⬇️ **Downloads** - JPG visualization and JSON report
- ⚙️ **Configuration** - Adjustable zoom levels and detection confidence

## How It Works

1. **Enter Coordinates** - Input latitude/longitude of parking lot
2. **Configure Settings** - Adjust zoom level (18-20) and confidence threshold
3. **Initialize Pipeline** - Load dual YOLO models (96.3% + 84% mAP50)
4. **Analyze** - Click "Analyze Parking Occupancy" button
5. **View Results** - See metrics, visualization, and download reports

## Processing Stages

1. 🗺️ **Downloading satellite imagery** - Google Maps API
2. 🔍 **Detecting parking stalls** - YOLOv11x model
3. 🚗 **Detecting vehicles** - YOLOv11m model
4. 📊 **Calculating occupancy** - Intersection over Union (IoU) analysis

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Maps API key (for satellite imagery)
- Internet connection

## Dependencies

See `requirements-streamlit.txt` for the full list. Core dependencies:

- streamlit >= 1.31.0
- PyTorch + torchvision
- OpenCV
- Ultralytics YOLO
- Pillow, numpy, pandas

## Configuration

Set your Google Maps API key:

```bash
export GOOGLE_MAPS_API_KEY="your_api_key_here"
```

Or create a `.env` file:

```
GOOGLE_MAPS_API_KEY=your_api_key_here
```

## Performance

- **Processing Time**: 30-60 seconds per location
- **Accuracy**: 96.3% mAP50 (stalls) + 84% mAP50 (vehicles)
- **Image Resolution**: 640x640 to 1280x1280 pixels (configurable)
- **Batch Processing**: Supports multiple locations

## Deployment

For production deployment, see `STREAMLIT_README.md` for:

- Docker deployment
- Cloud hosting (Streamlit Cloud, Heroku, AWS)
- Environment configuration
- Performance optimization

## Testing

```bash
python test_streamlit_pipeline.py
```

## Troubleshooting

- **Model Loading Issues**: Ensure models are in `weights/` directory
- **API Key Errors**: Check `GOOGLE_MAPS_API_KEY` environment variable
- **Memory Issues**: Reduce image resolution or use CPU mode
- **Import Errors**: Install missing dependencies from `requirements-streamlit.txt`

For detailed troubleshooting, see `STREAMLIT_README.md`.

## Support

For issues or questions:

1. Check `STREAMLIT_README.md` for technical documentation
2. Review `STREAMLIT_USER_GUIDE.md` for usage instructions
3. See main project README for architecture details

## License

Same as parent project - see main LICENSE file.
