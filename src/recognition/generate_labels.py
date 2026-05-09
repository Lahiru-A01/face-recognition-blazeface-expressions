import os
import csv
from pathlib import Path

PROCESSED_DIR = Path("data/processed_dataset")
OUTPUT_CSV    = Path("data/labels.csv")
MIN_IMAGES    = 5  # skip identities with fewer than 5 images

rows       = []
skipped_id = 0

identity_folders = sorted([f for f in PROCESSED_DIR.iterdir() if f.is_dir()])
print(f"Scanning {len(identity_folders)} identity folders...")

for label_id, folder in enumerate(identity_folders):
    images = [f for f in folder.iterdir()
              if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if len(images) < MIN_IMAGES:
        skipped_id += 1
        continue

    for img_path in images:
        rows.append({
            "image_path": str(img_path),
            "label":      label_id,
            "identity":   folder.name
        })

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "label", "identity"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{'='*50}")
print(f"✅ labels.csv created")
print(f"📊 Total images in CSV  : {len(rows)}")
print(f"👤 Total identities     : {len(set(r['label'] for r in rows))}")
print(f"⏭️  Skipped (< 5 images) : {skipped_id}")
print(f"📁 Saved to             : {OUTPUT_CSV}")
print(f"{'='*50}")