from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import cv2, numpy as np, os
from src.detection.blazeface_detector import BlazeFaceDetector
from src.preprocessing.face_align import align_face
from src.recognition.feature_extractor import FaceEmbedder
from src.utils.similarity import GalleryMatcher

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

THRESHOLD   = 0.50
MODEL_PATH  = "models/face_model_vggface2.pt"
GALLERY_DIR = "data/gallery"

os.makedirs(GALLERY_DIR, exist_ok=True)

detector = BlazeFaceDetector()
embedder = FaceEmbedder(MODEL_PATH)
matcher  = GalleryMatcher(GALLERY_DIR)

@app.get("/people")
async def get_people():
    files = [f.replace(".npy","") for f in os.listdir(GALLERY_DIR) if f.endswith(".npy")]
    return JSONResponse({"people": sorted(files)})

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"name": "NO FACE", "score": 0.0, "authorized": False})

    results = detector.detect(frame)
    if not results.detections:
        return JSONResponse({"name": "NO FACE", "score": 0.0, "authorized": False})

    detection = results.detections[0]
    aligned   = align_face(frame, detection)
    crop      = cv2.resize(aligned, (112, 112))
    embedding = embedder.get_embedding(crop)
    name, score = matcher.find_match(embedding, THRESHOLD)

    return JSONResponse({
        "name": name or "UNKNOWN",
        "score": float(score),
        "authorized": name is not None
    })

@app.post("/enroll/{person_name}")
async def enroll(person_name: str, file: UploadFile = File(...)):
    contents  = await file.read()
    nparr     = np.frombuffer(contents, np.uint8)
    frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"status": "invalid image"})

    results = detector.detect(frame)
    if not results.detections:
        return JSONResponse({"status": "no face detected — move closer"})

    detection = results.detections[0]
    aligned   = align_face(frame, detection)
    crop      = cv2.resize(aligned, (112, 112))
    embedding = embedder.get_embedding(crop)
    np.save(f"{GALLERY_DIR}/{person_name}.npy", embedding)
    matcher.load_gallery()
    return JSONResponse({"status": f"enrolled {person_name}"})

@app.delete("/remove/{person_name}")
async def remove(person_name: str):
    path = f"{GALLERY_DIR}/{person_name}.npy"
    if os.path.exists(path):
        os.remove(path)
        matcher.load_gallery()
        return JSONResponse({"status": f"removed {person_name}"})
    return JSONResponse({"status": "person not found"}, status_code=404)