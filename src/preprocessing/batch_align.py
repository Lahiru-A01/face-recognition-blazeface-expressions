import cv2
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.detection.blazeface_detector import BlazeFaceDetector
from src.preprocessing.face_align import align_face

INPUT_DIR  = ROOT / "data" / "raw_dataset" / "train"
OUTPUT_DIR = ROOT / "data" / "processed_dataset"
IMG_SIZE   = 112
SUPPORTED  = {".jpg", ".jpeg", ".png"}
PADDING    = 0.2

print("Loading BlazeFace detector...")
detector = BlazeFaceDetector(min_conf=0.5)
print("✅ Detector loaded.\n")

total = success = failed = skipped = 0

def crop_face(aligned_img, detection):
    h, w, _ = aligned_img.shape
    bbox = detection.location_data.relative_bounding_box
    x  = int((bbox.xmin - PADDING * bbox.width)  * w)
    y  = int((bbox.ymin - PADDING * bbox.height) * h)
    bw = int((bbox.width  + 2 * PADDING * bbox.width)  * w)
    bh = int((bbox.height + 2 * PADDING * bbox.height) * h)
    x  = max(0, x);  y  = max(0, y)
    bw = min(bw, w - x);  bh = min(bh, h - y)
    if bw <= 0 or bh <= 0:
        return None
    crop = aligned_img[y:y+bh, x:x+bw]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

identity_folders = sorted([f for f in INPUT_DIR.iterdir() if f.is_dir()])
print(f"Found {len(identity_folders)} identity folders.\n")

for identity_folder in tqdm(identity_folders, desc="Aligning identities"):
    out_identity = OUTPUT_DIR / identity_folder.name
    out_identity.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in identity_folder.iterdir()
                   if f.suffix.lower() in SUPPORTED]

    for img_file in image_files:
        total += 1
        out_path = out_identity / img_file.name

        if out_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            failed += 1
            continue

        try:
            results = detector.detect(img)
            if not results.detections:
                failed += 1
                continue

            detection = results.detections[0]
            aligned   = align_face(img, detection)

            aligned_results = detector.detect(aligned)
            if aligned_results.detections:
                crop = crop_face(aligned, aligned_results.detections[0])
            else:
                crop = crop_face(aligned, detection)

            if crop is None:
                failed += 1
                continue

            cv2.imwrite(str(out_path), crop)
            success += 1

        except Exception as e:
            failed += 1
            continue

print("\n" + "="*55)
print(f"✅ Successfully aligned  : {success}")
print(f"❌ Failed (no face/error): {failed}")
print(f"⏭️  Skipped (existed)     : {skipped}")
print(f"📊 Total processed       : {total}")
print(f"📁 Output saved to       : {OUTPUT_DIR}")
print(f"🎯 Success rate          : {success/(total-skipped)*100:.1f}%")
print("="*55)