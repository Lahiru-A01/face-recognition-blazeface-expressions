import numpy as np
import os

class GalleryMatcher:
    def __init__(self, gallery_dir="data/gallery"):
        self.gallery_dir = gallery_dir
        self.gallery     = {}
        os.makedirs(gallery_dir, exist_ok=True)
        self.load_gallery()

    def load_gallery(self):
        self.gallery = {}
        for f in os.listdir(self.gallery_dir):
            if f.endswith('.npy'):
                name = f.replace('.npy', '')
                emb  = np.load(f"{self.gallery_dir}/{f}")
                self.gallery[name] = emb
        print(f"✅ Gallery: {len(self.gallery)} people loaded")

    def save_embedding(self, name, embedding):
        path = f"{self.gallery_dir}/{name}.npy"
        np.save(path, embedding)
        self.gallery[name] = embedding
        print(f"✅ Saved: {name}")

    def find_match(self, embedding, threshold=0.5):
        if not self.gallery:
            return None, 0.0
        best_name  = None
        best_score = -1.0
        for name, gallery_emb in self.gallery.items():
            score = float(np.dot(embedding, gallery_emb))
            if score > best_score:
                best_score = score
                best_name  = name
        if best_score >= threshold:
            return best_name, best_score
        return None, best_score