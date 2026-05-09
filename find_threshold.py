import numpy as np
import cv2
import os
from src.recognition.feature_extractor import FaceEmbedder
from src.detection.blazeface_detector  import BlazeFaceDetector
from src.preprocessing.face_align      import align_face

embedder = FaceEmbedder("models/face_model_scripted.pt")
detector = BlazeFaceDetector()

gallery_dir = "data/gallery"
IMG_SIZE    = 112
PADDING     = 0.2

def get_face_embedding(frame):
    results = detector.detect(frame)
    if not results.detections:
        return None, 0.0
    detection = results.detections[0]
    aligned   = align_face(frame, detection)
    h, w, _   = aligned.shape
    bbox      = detection.location_data.relative_bounding_box
    x  = int((bbox.xmin - PADDING*bbox.width)  * w)
    y  = int((bbox.ymin - PADDING*bbox.height) * h)
    bw = int((bbox.width  + 2*PADDING*bbox.width)  * w)
    bh = int((bbox.height + 2*PADDING*bbox.height) * h)
    x  = max(0, x); y = max(0, y)
    bw = min(bw, w-x); bh = min(bh, h-y)
    if bw <= 0 or bh <= 0:
        return None, 0.0
    crop = aligned[y:y+bh, x:x+bw]
    if crop.size == 0:
        return None, 0.0
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return embedder.get_embedding(crop), detection

# Load your saved embedding
your_emb = np.load(f"{gallery_dir}/lahiru.npy")
print(f"✅ Loaded your embedding")
print(f"\n{'='*50}")
print(f"INSTRUCTIONS:")
print(f"  Phase 1 — Show YOUR face → press SPACE x10")
print(f"  Phase 2 — Show OTHER face → press SPACE x10")
print(f"  Press Q anytime to quit")
print(f"{'='*50}\n")

cap          = cv2.VideoCapture(0)
your_scores  = []
other_scores = []
phase        = 1  # 1=collecting yours, 2=collecting others
last_score   = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get embedding every frame
    emb, det = get_face_embedding(frame)
    if emb is not None:
        last_score = float(np.dot(emb, your_emb))

    # ── Draw UI ───────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 80), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if phase == 1:
        phase_text = f"Phase 1: Show YOUR face ({len(your_scores)}/10)"
        color      = (0, 255, 0)
    else:
        phase_text = f"Phase 2: Show OTHER person ({len(other_scores)}/10)"
        color      = (0, 0, 255)

    cv2.putText(frame, phase_text,
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)
    cv2.putText(frame, f"Score: {last_score:.4f} | SPACE=capture Q=quit",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 2)

    # Show face box
    if det:
        h, w, _ = frame.shape
        bbox = det.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

    cv2.imshow('Threshold Finder — Keep this window focused!', frame)

    # ── Key handling ──────────────────────────
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q') or key == 27:
        print("Quitting...")
        break

    elif key == ord(' '):
        if emb is None:
            print("⚠️  No face detected — try again")
            continue

        if phase == 1:
            your_scores.append(last_score)
            print(f"✅ YOUR  score {len(your_scores):2d}/10: {last_score:.4f}")
            if len(your_scores) >= 10:
                phase = 2
                print(f"\n{'='*40}")
                print(f"✅ Phase 1 done!")
                print(f"🔄 Now show a DIFFERENT person's face")
                print(f"   (use a photo on phone or ask someone)")
                print(f"{'='*40}\n")

        elif phase == 2:
            other_scores.append(last_score)
            print(f"❌ OTHER score {len(other_scores):2d}/10: {last_score:.4f}")
            if len(other_scores) >= 10:
                print("\n✅ Collection complete!")
                break

cap.release()
cv2.destroyAllWindows()

# ── Final Analysis ────────────────────────────
print(f"\n{'='*55}")
print(f"📊 THRESHOLD ANALYSIS REPORT")
print(f"{'='*55}")

if your_scores:
    avg_your  = np.mean(your_scores)
    min_your  = np.min(your_scores)
    max_your  = np.max(your_scores)
    print(f"YOUR  face:")
    print(f"  Scores  : {[f'{s:.3f}' for s in your_scores]}")
    print(f"  Average : {avg_your:.4f}")
    print(f"  Min     : {min_your:.4f}")
    print(f"  Max     : {max_your:.4f}")

if other_scores:
    avg_other = np.mean(other_scores)
    min_other = np.min(other_scores)
    max_other = np.max(other_scores)
    print(f"\nOTHER face:")
    print(f"  Scores  : {[f'{s:.3f}' for s in other_scores]}")
    print(f"  Average : {avg_other:.4f}")
    print(f"  Min     : {min_other:.4f}")
    print(f"  Max     : {max_other:.4f}")

if your_scores and other_scores:
    print(f"\n{'='*55}")
    gap = min_your - max_other
    print(f"Separation gap (min_yours - max_others): {gap:.4f}")

    if gap > 0:
        best = (min_your + max_other) / 2
        print(f"✅ PERFECT separation!")
        print(f"🎯 Best threshold : {best:.4f}")
        print(f"\n👉 Set in main.py: THRESHOLD = {best:.2f}")
    else:
        best = avg_your - 0.05
        print(f"⚠️  Some overlap — conservative threshold")
        print(f"🎯 Suggested threshold : {best:.4f}")
        print(f"\n👉 Set in main.py: THRESHOLD = {best:.2f}")

    print(f"{'='*55}")

elif your_scores:
    print(f"\nOnly YOUR scores collected.")
    print(f"Min YOUR score: {min_your:.4f}")
    print(f"👉 Temporarily set THRESHOLD = {min_your - 0.05:.2f}")
    print(f"   Run again with OTHER person to confirm")