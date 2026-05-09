import cv2
import os
import numpy as np
from src.detection.blazeface_detector  import BlazeFaceDetector
from src.preprocessing.face_align      import align_face
from src.recognition.feature_extractor import FaceEmbedder
from src.utils.similarity              import GalleryMatcher

# ── Config ────────────────────────────────────
THRESHOLD  = 0.88   # ORL model threshold
MODEL_PATH = "models/face_model_scripted.pt"
GALLERY_DIR= "data/gallery"
IMG_SIZE   = 112
GREEN      = (0, 255, 0)
RED        = (0, 0, 255)
WHITE      = (255, 255, 255)
YELLOW     = (0, 255, 255)

def crop_face(aligned, detection, pad=0.2):
    h, w, _ = aligned.shape
    bbox     = detection.location_data.relative_bounding_box
    x  = int((bbox.xmin - pad*bbox.width)  * w)
    y  = int((bbox.ymin - pad*bbox.height) * h)
    bw = int((bbox.width  + 2*pad*bbox.width)  * w)
    bh = int((bbox.height + 2*pad*bbox.height) * h)
    x  = max(0, x); y = max(0, y)
    bw = min(bw, w-x); bh = min(bh, h-y)
    if bw <= 0 or bh <= 0:
        return None
    crop = aligned[y:y+bh, x:x+bw]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

def draw_result(frame, detection, name, score, authorized):
    h, w, _ = frame.shape
    bbox     = detection.location_data.relative_bounding_box
    x1 = max(0, int(bbox.xmin * w))
    y1 = max(0, int(bbox.ymin * h))
    x2 = min(w, int((bbox.xmin + bbox.width)  * w))
    y2 = min(h, int((bbox.ymin + bbox.height) * h))
    color = GREEN if authorized else RED
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    label = f"{'AUTH: '+name if authorized else 'UNKNOWN'}  {score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+5, y1), color, -1)
    cv2.putText(frame, label, (x1+2, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    return frame

def main():
    print("🚀 Loading system...")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Place face_recognition_model.onnx in models/ folder")
        return

    detector = BlazeFaceDetector()
    embedder = FaceEmbedder(MODEL_PATH)
    matcher  = GalleryMatcher(GALLERY_DIR)
    cap      = cv2.VideoCapture(0)

    print("\n✅ System ready!")
    print("─" * 35)
    print("'s' → Enroll your face")
    print("'c' → Clear gallery")
    print("'q' → Quit")
    print("─" * 35)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = detector.detect(frame)

        if results.detections:
            for detection in results.detections:
                aligned = align_face(frame, detection)
                crop    = crop_face(aligned, detection)

                if crop is None:
                    continue

                embedding           = embedder.get_embedding(crop)
                name, score         = matcher.find_match(
                                        embedding, THRESHOLD)
                authorized          = name is not None
                frame               = draw_result(
                                        frame, detection,
                                        name or "UNKNOWN",
                                        score, authorized)

                cv2.imshow('Aligned Face', crop)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    save_name = input("Enter your name: ").strip()
                    if save_name:
                        matcher.save_embedding(save_name, embedding)
                elif key == ord('c'):
                    for f in os.listdir(GALLERY_DIR):
                        if f.endswith('.npy'):
                            os.remove(f"{GALLERY_DIR}/{f}")
                    matcher.load_gallery()
                    print("🗑️  Gallery cleared")
                elif key == ord('q'):
                    break

        status = f"Gallery: {len(matcher.gallery)} | Press 's' to enroll"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()