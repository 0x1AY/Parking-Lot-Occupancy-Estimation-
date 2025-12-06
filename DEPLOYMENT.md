# Streamlit Cloud Deployment Guide

## Important: Model Files Not Included

The trained YOLO model files (`.pt` files) are **too large** to be stored in Git and are **not included** in this repository.

### Required Model Files

You need these three model files:

1. **Localization Model** (Parking Lot Detection)
   - Path: `datasets/apklot/apklot_stage1/weights/best.pt`
   - Purpose: Detects parking lot areas in satellite images
   
2. **Car Detection Model** (96.3% mAP50)
   - Path: `parking_runs/yolo11m_parking_augmented2/weights/best.pt`
   - Purpose: High-accuracy vehicle detection
   
3. **Stall Detection Model** (84% mAP50)
   - Path: `parking_runs/yolo11m_multilabel/weights/best.pt`
   - Purpose: Parking stall localization

### Deployment Options

#### Option 1: Local Deployment (Recommended)
Run the app locally where you have the model files:
```bash
cd "/path/to/Parking-Lot-Occupancy-Estimation-"
streamlit run app/app.py
```

#### Option 2: Cloud Deployment with External Storage
For Streamlit Cloud deployment:

1. **Upload models to cloud storage** (Google Drive, Dropbox, AWS S3, etc.)

2. **Download models at runtime** - Add this to `app.py`:
```python
import gdown
import os

# Download models if not present
if not os.path.exists('datasets/apklot/apklot_stage1/weights/best.pt'):
    os.makedirs('datasets/apklot/apklot_stage1/weights', exist_ok=True)
    gdown.download(
        'https://drive.google.com/uc?id=YOUR_FILE_ID',
        'datasets/apklot/apklot_stage1/weights/best.pt'
    )
# Repeat for other models...
```

3. **Add to requirements-streamlit.txt**:
```
gdown>=4.7.1
```

#### Option 3: Use Streamlit Secrets
Store model URLs in Streamlit Cloud secrets and download on startup.

### Model File Sizes

These files are typically **100-500MB each** and cannot be committed to GitHub:
- GitHub has a 100MB file size limit
- Repository would become too large
- Files tracked by `.gitignore`: `*.pt`, `*.pth`, `*.weights`

### For Development Team

If you trained these models locally:
1. Keep model files in their current directories
2. Run the app locally
3. Do NOT commit `.pt` files to git
4. Share models via cloud storage links if needed

### Environment Setup

1. **Set API Key**:
```bash
export GOOGLE_MAPS_API_KEY="your_key_here"
```

2. **Install Dependencies**:
```bash
pip install -r app/requirements-streamlit.txt
```

3. **Run Application**:
```bash
streamlit run app/app.py
```

## Streamlit Cloud Configuration

### Files Required at Root:
- `packages.txt` - System dependencies (libgl1-mesa-glx, libglib2.0-0)
- `app/requirements-streamlit.txt` - Python dependencies
- `.streamlit/config.toml` (optional) - Streamlit configuration

### Environment Variables:
Set in Streamlit Cloud dashboard:
- `GOOGLE_MAPS_API_KEY` - Your Google Maps API key

### Known Issues:
- **ImportError: libGL.so.1** - Fixed by `packages.txt`
- **Model not found** - Models need to be downloaded separately
- **Large file size** - Use cloud storage for model files

## Support

For questions about deployment:
- Check app/STREAMLIT_README.md for technical details
- Review main README.md for project architecture
- Contact: Northeastern University Deep Learning Team
