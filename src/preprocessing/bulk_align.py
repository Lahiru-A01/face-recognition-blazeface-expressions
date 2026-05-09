import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from src.detection.blazeface_det import BlazeFaceDetector
from src.preprocessing.face_align import FaceAligner

# CONFIGURATION
RAW_DIR = "C:/Datasets/CASIA-WebFace"
ALIGNED_DIR = "C:/Datasets/CASIA_Aligned"
THREADS = 8 # Adjust based on your CPU cores

detector = BlazeFaceDetector()
aligner = FaceAligner(output_size=(112, 112))

def process_image(rel_path):
    input_path = os.path.join(RAW_DIR, rel_path)
    output_path = os.path.join(ALIGNED_DIR, rel_path)
    
    # Ensure the person's identity folder exists in the output dir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = cv2.imread(input_path)
    if img is None: return False

    # 1. Detection & Filtering (Skip if no face found)
    landmarks = detector.extract_landmarks(img)
    if landmarks is None or len(landmarks) == 0:
        return False

    # 2. Affine Alignment & Cropping
    face_chip = aligner.align(img, landmarks)

    # 3. Save as PNG for lossless training quality
    cv2.imwrite(output_path, face_chip)
    return True

if __name__ == "__main__":
    # Get all image paths
    tasks = []
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                tasks.append(os.path.relpath(os.path.join(root, f), RAW_DIR))

    print(f"Aligning {len(tasks)} images...")
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        list(tqdm(executor.map(process_image, tasks), total=len(tasks)))