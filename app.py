#!/usr/bin/env python3
"""
Streamlit Web App for Parking Occupancy Detection
=================================================
Real-time parking occupancy detection from satellite imagery.

Usage:
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import json
import tempfile
import time
from datetime import datetime

# Add occupancy directory to path
sys.path.insert(0, str(Path(__file__).parent / "occupancy"))
from unified_parking_pipeline import UnifiedParkingPipeline

# Page configuration
st.set_page_config(
    page_title="Parking Occupancy Detection",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Header
st.markdown('<div class="main-header">🅿️ Parking Occupancy Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time parking occupancy analysis using satellite imagery and deep learning</div>', unsafe_allow_html=True)

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Model Settings")
    
    # Model paths
    localization_model = st.text_input(
        "Localization Model",
        value="datasets/apklot/apklot_stage1/weights/best.pt",
        help="Path to parking lot localization model"
    )
    
    car_model = st.text_input(
        "Car Detection Model",
        value="parking_runs/yolo11m_parking_augmented2/weights/best.pt",
        help="High-accuracy car detection model (96.3% mAP50)"
    )
    
    stall_model = st.text_input(
        "Stall Detection Model",
        value="parking_runs/yolo11m_multilabel/weights/best.pt",
        help="Multiclass stall detection model (84% mAP50)"
    )
    
    api_key = st.text_input(
        "Google Maps API Key",
        value="AIzaSyCZWUlRCSb7WxHNBWtMifWRW25GOWfbous",
        type="password",
        help="Your Google Static Maps API key"
    )
    
    st.divider()
    
    st.subheader("Detection Parameters")
    
    localization_zoom = st.slider(
        "Localization Zoom",
        min_value=17,
        max_value=20,
        value=19,
        help="Zoom level for initial parking lot detection"
    )
    
    tile_zoom = st.slider(
        "Tile Zoom",
        min_value=19,
        max_value=21,
        value=20,
        help="Zoom level for high-resolution tile downloads"
    )
    
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="IoU threshold for car-to-stall matching"
    )
    
    st.divider()
    
    # Initialize button
    if st.button("🔄 Initialize Pipeline", type="primary", use_container_width=True):
        with st.spinner("Loading models..."):
            try:
                st.session_state.pipeline = UnifiedParkingPipeline(
                    localization_model_path=localization_model,
                    car_model_path=car_model,
                    stall_model_path=stall_model,
                    google_api_key=api_key
                )
                st.success("✅ Pipeline initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize pipeline: {str(e)}")

# Main content area
if st.session_state.pipeline is None:
    st.markdown('<div class="info-box">👈 Please initialize the pipeline using the sidebar settings.</div>', unsafe_allow_html=True)
    
    # Display system information
    st.subheader("📊 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Dual-Model Architecture**
        - Car Detection: 96.3% mAP50
        - Stall Detection: 84% mAP50
        - IoU-based matching algorithm
        """)
    
    with col2:
        st.markdown("""
        **🔄 4-Stage Pipeline**
        1. Parking lot localization (Z19)
        2. High-res tile download (Z20)
        3. Object detection (dual models)
        4. Occupancy calculation
        """)
    
    with col3:
        st.markdown("""
        **📈 Proven Results**
        - 10 locations validated
        - 813 stalls detected
        - 27.8% avg occupancy
        - 100% success rate
        """)
    
    st.divider()
    
    # Example locations
    st.subheader("📍 Example Locations")
    
    examples = {
        "Walmart Gerrard St": {"lat": 43.668734, "lon": -79.340158},
        "Walmart Dufferin St": {"lat": 43.666156, "lon": -79.444583},
        "Walmart St Clair Ave": {"lat": 43.675844, "lon": -79.505278},
        "Walmart Islington Ave": {"lat": 43.665417, "lon": -79.583611},
        "Walmart Lawrence Ave": {"lat": 43.712778, "lon": -79.473333}
    }
    
    cols = st.columns(len(examples))
    for idx, (name, coords) in enumerate(examples.items()):
        with cols[idx]:
            st.markdown(f"**{name}**")
            st.caption(f"Lat: {coords['lat']:.6f}")
            st.caption(f"Lon: {coords['lon']:.6f}")

else:
    # Pipeline is initialized - show input form
    st.markdown('<div class="success-box">✅ Pipeline ready! Enter coordinates below to analyze parking occupancy.</div>', unsafe_allow_html=True)
    
    # Input form
    st.subheader("📍 Location Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=43.668734,
            format="%.6f",
            help="Latitude of parking lot center"
        )
    
    with col2:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-79.340158,
            format="%.6f",
            help="Longitude of parking lot center"
        )
    
    location_name = st.text_input(
        "Location Name (Optional)",
        value="",
        placeholder="e.g., Walmart Gerrard St",
        help="Optional name for this location"
    )
    
    # Process button
    if st.button("🚀 Analyze Parking Occupancy", type="primary", use_container_width=True):
        if not location_name:
            location_name = f"location_{latitude:.4f}_{longitude:.4f}"
        
        st.session_state.processing = True
        st.session_state.results = None
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            output_dir.mkdir(exist_ok=True)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Stage 1: Localization
                status_text.text("🔍 Stage 1/4: Detecting parking lot areas...")
                progress_bar.progress(0.25)
                time.sleep(0.5)
                
                # Stage 2: Download tiles
                status_text.text("📥 Stage 2/4: Downloading high-resolution tiles...")
                progress_bar.progress(0.50)
                time.sleep(0.5)
                
                # Stage 3: Detection
                status_text.text("🚗 Stage 3/4: Detecting cars and parking stalls...")
                progress_bar.progress(0.75)
                
                # Process location
                start_time = time.time()
                results = st.session_state.pipeline.process_location(
                    location_name=location_name,
                    lat=latitude,
                    lon=longitude,
                    output_dir=output_dir,
                    localization_zoom=localization_zoom,
                    tile_zoom=tile_zoom,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                processing_time = time.time() - start_time
                
                # Stage 4: Complete
                status_text.text("✅ Stage 4/4: Processing complete!")
                progress_bar.progress(1.0)
                
                # Store results
                st.session_state.results = results
                st.session_state.processing_time = processing_time
                
                # Clear progress indicators
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)
                st.session_state.processing = False
                progress_bar.empty()
                status_text.empty()
    
    # Display results if available
    if st.session_state.results is not None:
        results = st.session_state.results
        
        st.divider()
        st.subheader("📊 Analysis Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Stalls",
                value=results.get('total_stalls', 0),
                help="Total number of parking stalls detected"
            )
        
        with col2:
            st.metric(
                label="Occupied Stalls",
                value=results.get('occupied_stalls', 0),
                help="Number of stalls with cars detected"
            )
        
        with col3:
            occupancy_rate = results.get('occupancy_rate', 0)
            st.metric(
                label="Occupancy Rate",
                value=f"{occupancy_rate:.1f}%",
                help="Percentage of occupied parking stalls"
            )
        
        with col4:
            st.metric(
                label="Processing Time",
                value=f"{st.session_state.processing_time:.1f}s",
                help="Total time to process this location"
            )
        
        # Visualization
        st.divider()
        st.subheader("🗺️ Occupancy Visualization")
        
        # Check if visualization image exists
        result_path = results.get('result_path')
        if result_path and Path(result_path).exists():
            occupancy_img = Image.open(result_path)
            st.image(occupancy_img, caption="Parking Occupancy Map (Green=Vacant, Red=Occupied)", use_container_width=True)
        else:
            st.warning("⚠️ Visualization image not found")
        
        # Detailed metrics
        st.divider()
        st.subheader("📈 Detailed Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Detection Statistics**")
            detection_stats = {
                "Cars Detected": results.get('cars_detected', 0),
                "Stalls Detected": results.get('total_stalls', 0),
                "Occupied Stalls": results.get('occupied_stalls', 0),
                "Vacant Stalls": results.get('total_stalls', 0) - results.get('occupied_stalls', 0),
            }
            for key, value in detection_stats.items():
                st.markdown(f"- **{key}**: {value}")
        
        with col2:
            st.markdown("**Location Details**")
            location_info = {
                "Latitude": f"{latitude:.6f}",
                "Longitude": f"{longitude:.6f}",
                "Location Name": location_name,
                "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            for key, value in location_info.items():
                st.markdown(f"- **{key}**: {value}")
        
        # JSON output
        with st.expander("📄 View Raw JSON Output"):
            st.json(results)
        
        # Download button for results
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if result_path and Path(result_path).exists():
                with open(result_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Visualization",
                        data=f,
                        file_name=f"{location_name}_occupancy.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
        
        with col2:
            json_data = json.dumps(results, indent=2)
            st.download_button(
                label="⬇️ Download JSON Report",
                data=json_data,
                file_name=f"{location_name}_report.json",
                mime="application/json",
                use_container_width=True
            )

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <p><strong>Parking Occupancy Detection System</strong> | Powered by YOLOv11 Dual-Model Architecture</p>
    <p>Car Detection: 96.3% mAP50 | Stall Detection: 84% mAP50</p>
    <p>© 2025 Northeastern University | Deep Learning Project</p>
</div>
""", unsafe_allow_html=True)
