import torch
import numpy as np
import cv2

class FaceEmbedder:
    def __init__(self, model_path="models/face_model_scripted.pt"):
        # Load TorchScript model on CPU
        self.model    = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()
        self.img_size = 112
        print(f"✅ FaceEmbedder loaded: {model_path}")

    def preprocess(self, face_img):
        """Convert face crop to model input tensor."""
        img = cv2.resize(face_img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img)

    def get_embedding(self, face_img):
        """Extract 128D embedding from face crop."""
        tensor = self.preprocess(face_img)
        with torch.no_grad():
            embedding = self.model(tensor).numpy()[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def cosine_similarity(self, emb1, emb2):
        return float(np.dot(emb1, emb2))