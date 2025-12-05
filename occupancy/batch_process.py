#!/usr/bin/env python3
"""
Batch process all Walmart locations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from occupancy.unified_parking_pipeline import UnifiedParkingPipeline
import json

# Walmart locations with their coordinates
WALMART_LOCATIONS = [
    {
        'name': 'walmart_01_1000_Gerrard_St_E_Toronto_ON_M4M_0A5',
        'lat': 43.668734,
        'lon': -79.340158
    },
    {
        'name': 'walmart_02_900_Dufferin_St_Toronto_ON_M6H_4A9',
        'lat': 43.671667,
        'lon': -79.436389
    },
    {
        'name': 'walmart_03_2525_St_Clair_Ave_W_Toronto_ON_M6N_4Z5',
        'lat': 43.674722,
        'lon': -79.491667
    },
    {
        'name': 'walmart_04_165_N_Queen_St_Toronto_ON_M9C_1A7',
        'lat': 43.639722,
        'lon': -79.568056
    },
    {
        'name': 'walmart_05_2245_Islington_Ave_Toronto_ON_M9W_3W6',
        'lat': 43.687222,
        'lon': -79.564167
    },
    {
        'name': 'walmart_06_1500_Dundas_St_E_Mississauga_ON_L4X_1L4',
        'lat': 43.611111,
        'lon': -79.609722
    },
    {
        'name': 'walmart_07_1305_Lawrence_Ave_W_Toronto_ON_M6L_1A5',
        'lat': 43.714167,
        'lon': -79.486111
    },
    {
        'name': 'walmart_08_1900_Eglinton_Ave_E_Scarborough_ON_M1L_2L9',
        'lat': 43.727778,
        'lon': -79.268333
    },
    {
        'name': 'walmart_09_2202_Jane_St_North_York_ON_M3M_1A4',
        'lat': 43.754167,
        'lon': -79.504167
    },
    {
        'name': 'walmart_10_3757_Keele_St_Toronto_ON_M3J_1N4',
        'lat': 43.758333,
        'lon': -79.477222
    }
]


def main():
    print("="*70)
    print("BATCH PROCESSING ALL WALMART LOCATIONS")
    print("="*70)
    
    # Initialize pipeline with dual-model architecture
    pipeline = UnifiedParkingPipeline(
        car_model_path="parking_runs/yolo11m_parking_augmented2/weights/best.pt",
        stall_model_path="parking_runs/yolo11m_multilabel/weights/best.pt"
    )
    
    results_summary = []
    
    for idx, location in enumerate(WALMART_LOCATIONS, 1):
        print(f"\n\n{'='*70}")
        print(f"LOCATION {idx}/10: {location['name']}")
        print(f"{'='*70}\n")
        
        # Find image file
        image_pattern = f"walmart_locations/wide_area_z19/{location['name']}_z19_640x640-2x.png"
        image_path = Path(image_pattern)
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            results_summary.append({
                'location': location['name'],
                'status': 'failed',
                'reason': 'image not found'
            })
            continue
        
        try:
            # Run pipeline
            result = pipeline.run_pipeline(
                image_path,
                location['lat'],
                location['lon'],
                zoom=19,
                output_dir=None,  # Will use default
                conf_stage1=0.7,
                conf_stage3=0.25
            )
            
            if result:
                results_summary.append({
                    'location': location['name'],
                    'status': 'success',
                    'occupancy_rate': result['summary']['occupancy_rate'],
                    'total_stalls': result['summary']['total_stalls'],
                    'occupied_stalls': result['summary']['occupied_stalls'],
                    'empty_stalls': result['summary']['empty_stalls']
                })
                print(f"\n✓ SUCCESS: {result['summary']['occupancy_rate']}% occupancy")
            else:
                results_summary.append({
                    'location': location['name'],
                    'status': 'failed',
                    'reason': 'no parking areas detected'
                })
                print(f"\n❌ FAILED: No parking areas detected")
                
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            results_summary.append({
                'location': location['name'],
                'status': 'error',
                'reason': str(e)
            })
    
    # Save batch summary
    summary_path = Path('occupancy/results/batch_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print summary
    print("\n\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    
    successful = sum(1 for r in results_summary if r['status'] == 'success')
    failed = len(results_summary) - successful
    
    print(f"\nProcessed: {len(results_summary)} locations")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print("\nOccupancy rates:")
        for result in results_summary:
            if result['status'] == 'success':
                print(f"  {result['location'][:30]:30s}: {result['occupancy_rate']:6.2f}% "
                      f"({result['occupied_stalls']}/{result['total_stalls']} stalls)")
    
    print(f"\nResults saved to: {summary_path}")
    print(f"Individual outputs in: occupancy/results/")


if __name__ == '__main__':
    main()
