"""
Quick Demo Script - Test the Streamlit App Locally
===================================================

This script demonstrates how to test the Streamlit app programmatically
before running the full web interface.
"""

from pathlib import Path
import sys

# Add occupancy to path
sys.path.insert(0, str(Path(__file__).parent / "occupancy"))
from unified_parking_pipeline import UnifiedParkingPipeline

def test_pipeline():
    """Test the pipeline with a sample location."""
    
    print("="*70)
    print("Testing Unified Parking Pipeline")
    print("="*70)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    try:
        pipeline = UnifiedParkingPipeline(
            localization_model_path="datasets/apklot/apklot_stage1/weights/best.pt",
            car_model_path="parking_runs/yolo11m_parking_augmented2/weights/best.pt",
            stall_model_path="parking_runs/yolo11m_multilabel/weights/best.pt"
        )
        print("✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return False
    
    # Test location
    print("\n2. Testing with sample location (Walmart Gerrard St)...")
    test_lat = 43.668734
    test_lon = -79.340158
    output_dir = Path("test_output")
    
    try:
        results = pipeline.process_location(
            location_name="test_walmart",
            lat=test_lat,
            lon=test_lon,
            output_dir=output_dir,
            localization_zoom=19,
            tile_zoom=20,
            conf_threshold=0.25,
            iou_threshold=0.3
        )
        
        print("\n✓ Processing complete!")
        print("\nResults:")
        print(f"  Location: {results['location_name']}")
        print(f"  Coordinates: ({results['latitude']:.6f}, {results['longitude']:.6f})")
        print(f"  Total Stalls: {results['total_stalls']}")
        print(f"  Occupied: {results['occupied_stalls']}")
        print(f"  Occupancy Rate: {results['occupancy_rate']:.1f}%")
        print(f"  Cars Detected: {results['cars_detected']}")
        print(f"  Result Path: {results['result_path']}")
        
        if results['result_path'] and Path(results['result_path']).exists():
            print(f"\n✓ Visualization saved: {results['result_path']}")
            print(f"  View it with: open {results['result_path']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("\n🅿️  Parking Occupancy Detection - Pipeline Test")
    print()
    
    # Check if models exist
    print("Checking models...")
    models = {
        "Localization": "datasets/apklot/apklot_stage1/weights/best.pt",
        "Car Detection": "parking_runs/yolo11m_parking_augmented2/weights/best.pt",
        "Stall Detection": "parking_runs/yolo11m_multilabel/weights/best.pt"
    }
    
    all_present = True
    for name, path in models.items():
        if Path(path).exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (NOT FOUND)")
            all_present = False
    
    if not all_present:
        print("\n❌ Some models are missing. Please ensure all trained models are present.")
        return
    
    print("\n✓ All models found!")
    
    # Run test
    success = test_pipeline()
    
    if success:
        print("\n" + "="*70)
        print("✅ Test completed successfully!")
        print("="*70)
        print("\nYou can now run the Streamlit app:")
        print("  ./run_app.sh")
        print("  or")
        print("  streamlit run app.py")
    else:
        print("\n" + "="*70)
        print("❌ Test failed. Please check the error messages above.")
        print("="*70)

if __name__ == "__main__":
    main()
