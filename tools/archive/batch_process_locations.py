#!/usr/bin/env python3
"""
Batch process all Walmart locations for occupancy analysis.
"""

import subprocess
from pathlib import Path
import json
import csv
from datetime import datetime

# Walmart locations with coordinates
LOCATIONS = [
    {
        'name': 'walmart_01_1000_Gerrard_St_E_Toronto_ON_M4M_0A5',
        'lat': 43.668734,
        'lon': -79.340158
    },
    {
        'name': 'walmart_02_1_Bass_Pro_Mills_Dr_Vaughan_ON_L4K_5W4',
        'lat': 43.826523,
        'lon': -79.538208
    },
    {
        'name': 'walmart_03_8190_ON-27_Vaughan_ON_L4L_1A6',
        'lat': 43.786819,
        'lon': -79.563080
    },
    {
        'name': 'walmart_04_90_Riocan_Ave_North_York_ON_M9M_0A',
        'lat': 43.742142,
        'lon': -79.554108
    },
    {
        'name': 'walmart_05_150_Elgin_Mills_Rd_E_Richmond_Hill_ON_L4S_0B1',
        'lat': 43.890385,
        'lon': -79.413673
    },
    {
        'name': 'walmart_06_1900_Eglinton_Ave_E_Scarborough_ON_M1L_2L9',
        'lat': 43.724014,
        'lon': -79.281166
    },
    {
        'name': 'walmart_07_1240_Bay_St_Toronto_ON_M5R_2A7',
        'lat': 43.672508,
        'lon': -79.390129
    },
    {
        'name': 'walmart_08_4141_Dixie_Rd_Mississauga_ON_L4W_1V5',
        'lat': 43.593224,
        'lon': -79.617241
    },
    {
        'name': 'walmart_09_150_Elgin_Mills_Rd_E_Richmond_Hill_ON_L4S_0B2',
        'lat': 43.890095,
        'lon': -79.413208
    },
    {
        'name': 'walmart_10_3045_Mavis_Rd_Mississauga_ON_L5B_4M8',
        'lat': 43.599926,
        'lon': -79.650360
    }
]


def process_location(location, base_dir, conf_stage1=0.7, conf_stage3=0.25):
    """Process a single location."""
    print(f"\n{'='*70}")
    print(f"Processing: {location['name']}")
    print(f"{'='*70}")
    
    # Find image file
    image_pattern = f"{location['name']}_z19_640x640-2x.png"
    image_path = base_dir / image_pattern
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Run pipeline
    cmd = [
        'python', 'tools/parking_detection_pipeline.py',
        '--image', str(image_path),
        '--lat', str(location['lat']),
        '--lon', str(location['lon']),
        '--zoom', '19',
        '--conf-stage1', str(conf_stage1),
        '--conf-stage3', str(conf_stage3)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Read results
        results_dir = image_path.parent / 'pipeline_results'
        json_path = results_dir / 'overall_occupancy.json'
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                return {
                    'location': location['name'],
                    'lat': location['lat'],
                    'lon': location['lon'],
                    **data['summary']
                }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {location['name']}")
        print(e.stderr)
        return None
    
    return None


def main():
    base_dir = Path('walmart_locations/wide_area_z19')
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return
    
    print("="*70)
    print("BATCH PROCESSING ALL WALMART LOCATIONS")
    print("="*70)
    print(f"Total locations: {len(LOCATIONS)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process all locations
    all_results = []
    
    for i, location in enumerate(LOCATIONS, 1):
        print(f"\n[{i}/{len(LOCATIONS)}]")
        result = process_location(location, base_dir)
        
        if result:
            all_results.append(result)
            print(f"✓ Success: {result['occupancy_rate']}% occupancy")
        else:
            print(f"⚠️ Skipped or failed")
    
    # Generate summary report
    if all_results:
        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        print(f"\nProcessed {len(all_results)}/{len(LOCATIONS)} locations successfully")
        
        # Calculate overall statistics
        total_stalls = sum(r['total_stalls'] for r in all_results)
        total_occupied = sum(r['occupied_stalls'] for r in all_results)
        total_empty = sum(r['empty_stalls'] for r in all_results)
        avg_occupancy = sum(r['occupancy_rate'] for r in all_results) / len(all_results)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total stalls across all locations: {total_stalls}")
        print(f"  Total occupied: {total_occupied}")
        print(f"  Total empty: {total_empty}")
        print(f"  Average occupancy rate: {avg_occupancy:.1f}%")
        
        # Save summary CSV
        summary_path = Path('walmart_locations/batch_summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'location', 'lat', 'lon', 'total_stalls', 'occupied_stalls', 
                'empty_stalls', 'total_cars', 'unmatched_cars', 'occupancy_rate'
            ])
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n✓ Summary saved: {summary_path}")
        
        # Display table
        print(f"\nDETAILED RESULTS:")
        print(f"{'Location':<50} {'Stalls':>8} {'Occupied':>10} {'Occupancy':>10}")
        print("-" * 80)
        for r in all_results:
            loc_name = r['location'].replace('walmart_', '').replace('_', ' ')[:47]
            print(f"{loc_name:<50} {r['total_stalls']:>8} {r['occupied_stalls']:>10} {r['occupancy_rate']:>9.1f}%")


if __name__ == '__main__':
    main()
