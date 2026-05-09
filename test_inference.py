import cv2
import numpy as np
import torch
from src.recognition.feature_extractor import FaceEmbedder

print("🔍 Testing TorchScript model...")

# Load model
embedder = FaceEmbedder("models/face_model_scripted.pt")
print("✅ Model loaded!")

# Test 1 — Two random inputs give different embeddings
dummy1 = np.random.randint(0, 255, (112,112,3), dtype=np.uint8)
dummy2 = np.random.randint(0, 255, (112,112,3), dtype=np.uint8)

emb1 = embedder.get_embedding(dummy1)
emb2 = embedder.get_embedding(dummy2)

sim = float(np.dot(emb1, emb2))
print(f"✅ Embedding shape : {emb1.shape}")
print(f"✅ Embedding norm  : {np.linalg.norm(emb1):.4f}")
print(f"✅ Random sim      : {sim:.4f}")

if abs(sim) < 0.99:
    print("✅ Model working correctly!")
else:
    print("❌ Model collapsed!")

# Test 2 — Gallery image if exists
import os
gallery = "data/gallery/me_aligned.jpg"
if os.path.exists(gallery):
    img = cv2.imread(gallery)
    emb = embedder.get_embedding(img)
    print(f"\n✅ Gallery image embedding: {emb[:4]}")

print("\n🎉 Ready to run main.py!")