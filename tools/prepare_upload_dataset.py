#!/usr/bin/env python3
"""
Prepare Dataset-V1-detect for upload by copying actual images instead of symlinks.
Creates Dataset-V1-detect-upload with real image files.
"""

import shutil
from pathlib import Path

# Paths
base_dir = Path(__file__).parent.parent
source_dataset = base_dir / "Dataset-V1-detect"
upload_dataset = base_dir / "Dataset-V1-detect-upload"

print("📦 Preparing Dataset-V1-detect for Upload")
print("=" * 70)
print(f"Source: {source_dataset}")
print(f"Destination: {upload_dataset}")
print()

# Create upload directory
if upload_dataset.exists():
    print("⚠️  Upload directory already exists. Removing...")
    shutil.rmtree(upload_dataset)

upload_dataset.mkdir(parents=True)

# Copy data.yaml
print("📄 Copying data.yaml...")
shutil.copy2(source_dataset / "data.yaml", upload_dataset / "data.yaml")

# Copy README files if they exist
for readme in ["README.dataset.txt", "README.roboflow.txt"]:
    readme_path = source_dataset / readme
    if readme_path.exists():
        print(f"📄 Copying {readme}...")
        shutil.copy2(readme_path, upload_dataset / readme)

# Process each split
for split in ['train', 'valid', 'test']:
    print(f"\n📂 Processing {split}/ split...")
    
    source_split = source_dataset / split
    dest_split = upload_dataset / split
    
    if not source_split.exists():
        print(f"  ⚠️  {split}/ not found, skipping...")
        continue
    
    # Create split directories
    (dest_split / "images").mkdir(parents=True, exist_ok=True)
    (dest_split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Copy labels
    source_labels = source_split / "labels"
    dest_labels = dest_split / "labels"
    
    if source_labels.exists():
        label_files = list(source_labels.glob("*.txt"))
        print(f"  📝 Copying {len(label_files)} label files...")
        for label_file in label_files:
            shutil.copy2(label_file, dest_labels / label_file.name)
    
    # Copy or resolve images
    source_images = source_split / "images"
    dest_images = dest_split / "images"
    
    if source_images.exists():
        if source_images.is_symlink():
            # Resolve symlink and copy actual images
            actual_images_dir = source_images.resolve()
            print(f"  🔗 Resolving symlink: {actual_images_dir}")
            image_files = list(actual_images_dir.glob("*"))
            image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            print(f"  🖼️  Copying {len(image_files)} image files...")
            
            for idx, image_file in enumerate(image_files, 1):
                shutil.copy2(image_file, dest_images / image_file.name)
                if idx % 20 == 0:
                    print(f"     Copied {idx}/{len(image_files)}...")
        else:
            # Regular directory, copy all images
            image_files = list(source_images.glob("*"))
            image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            print(f"  🖼️  Copying {len(image_files)} image files...")
            
            for idx, image_file in enumerate(image_files, 1):
                shutil.copy2(image_file, dest_images / image_file.name)
                if idx % 20 == 0:
                    print(f"     Copied {idx}/{len(image_files)}...")

print("\n" + "=" * 70)
print("✅ Dataset preparation complete!")
print(f"\n📦 Upload-ready dataset: {upload_dataset}")
print("\nYou can now upload Dataset-V1-detect-upload/ to GitHub, Google Drive, or Kaggle.")
print("It contains actual image files instead of symlinks.")
