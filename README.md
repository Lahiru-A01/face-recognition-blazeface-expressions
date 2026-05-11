# Expression-Invariant Face Recognition System

Real-time face authentication system robust to head tilts and facial expressions using BlazeFace + MobileFaceNet + ArcFace loss, trained on VGGFace2.

🌐 **Live Demo: [https://faze-dr.onrender.com/static/index.html](https://faze-dr.onrender.com/static/index.html)**

---

## Features

- Real-time face detection (BlazeFace / MediaPipe)
- Geometric alignment via Affine Transformation
- Custom MobileFaceNet backbone with ArcFace Loss
- 128D L2-normalized face embeddings
- CPU inference — no GPU needed at runtime
- Admin-protected enrollment panel
- Web-based interface — works in any browser
- Enroll new person in seconds

---

## Performance

| Metric | Value |
|---|---|
| Training dataset | VGGFace2 (176,330 images, 480 identities) |
| Training epochs | 30 |
| Final loss | 1.36 |
| Same-person similarity | 0.71 |
| Cross-person similarity | -0.08 |
| Separation gap | 0.80 |
| Accuracy (100-pair test) | ~81% |
| Stranger rejection | ~95% |

---

## Architecture

```
Webcam → BlazeFace Detection → Affine Alignment → MobileFaceNet (128D embedding) → Cosine Similarity → Identity
```

### Model Details
- **Backbone**: MobileFaceNet (~600k parameters)
- **Loss**: ArcFace (s=32, m=0.2)
- **Embedding**: 128D L2-normalized
- **Input**: 112×112 aligned face crops
- **Optimizer**: AdamW (lr=3e-4, cosine annealing)

---

## Web App (Deployed)

The system is deployed as a FastAPI web service with a professional dark UI.

**URL**: [https://faze-dr.onrender.com/static/index.html](https://faze-dr.onrender.com/static/index.html)

### How to use
1. Open the link in any browser
2. Allow camera access
3. Click **ADMIN** → enter password → enroll your face
4. Recognition runs automatically every 1.2 seconds
5. Green box = authorized, Red box = unknown

---

## Local Setup

### 1. Clone repo
```bash
git clone https://github.com/Lahiru-A01/face-recognition-blazeface-expressions
cd face-recognition-blazeface-expressions
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run desktop app
```bash
python main.py
```

**Controls:**
- `s` → Enroll your face
- `c` → Clear gallery
- `q` → Quit

### 5. Run web app locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.

---

## Training

The model was trained on Google Colab T4 GPU.

- **Dataset**: VGGFace2 — 176,330 geometrically aligned images across 480 identities
- **Notebook**: `notebooks/Colab_Training.ipynb`
- **Duration**: ~3.6 hours (30 epochs)

### Training config
```
Batch size  : 64
Optimizer   : AdamW (lr=3e-4)
Scheduler   : CosineAnnealingLR
ArcFace     : s=32, m=0.2
Embedding   : 128D
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Face detection | MediaPipe BlazeFace |
| Face recognition | PyTorch MobileFaceNet |
| Loss function | ArcFace |
| Web framework | FastAPI |
| Deployment | Render (Docker) |
| Frontend | HTML/CSS/JS |
| Runtime | Python 3.12, CPU only |

---

## Project Structure

```
face-recognition-blazeface-expressions/
├── app.py                  ← FastAPI web server
├── main.py                 ← Desktop app (OpenCV)
├── Dockerfile              ← Container config
├── requirements.txt        ← Dependencies
├── models/
│   └── face_model_vggface2.pt  ← Trained model
├── static/
│   └── index.html          ← Web UI
├── data/
│   └── gallery/            ← Enrolled face embeddings (.npy)
├── src/
│   ├── detection/          ← BlazeFace detector
│   ├── preprocessing/      ← Face alignment
│   ├── recognition/        ← MobileFaceNet embedder
│   └── utils/              ← Similarity matching
└── notebooks/
    └── Colab_Training.ipynb
```

---

## Author

**Lahiru** — [@Lahiru-A01](https://github.com/Lahiru-A01)