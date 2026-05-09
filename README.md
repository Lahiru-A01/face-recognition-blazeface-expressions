#  Expression-Invariant Face Recognition System

Real-time face authentication system robust to head tilts 
and facial expressions using BlazeFace + MobileFaceNet + ArcFace.

##  Features
- Real-time face detection (BlazeFace)
- Geometric alignment via Affine Transformation
- Custom MobileFaceNet with ArcFace Loss
- 128D face embeddings
- CPU inference (no GPU needed)
- Enroll new person in 5 seconds

##  Performance
- Training accuracy: 95% (ORL dataset)
- Separation gap: 0.72
- Real-time: 30+ FPS on CPU

##  Setup

### 1. Clone repo
git clone https://github.com/Lahiru-A01/face-recognition-blazeface-expressions
cd face-recognition-blazeface-expressions

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

### 3. Install dependencies
pip install -r requirements.txt

### 4. Download model
Download face_model_scripted.pt from releases
Place in models/ folder

### 5. Run
python main.py

##  Controls
- s → Enroll your face
- c → Clear gallery
- q → Quit

##  Architecture
BlazeFace → Affine Alignment → MobileFaceNet → ArcFace → Cosine Similarity

##  Training
See notebooks/Colab_Training.ipynb for full training pipeline
Dataset: ORL Face Database / VGGFace2

##  Tech Stack
- Python 3.11
- PyTorch
- OpenCV
- MediaPipe BlazeFace
- ONNX Runtime