import os
import json
import pickle
import numpy as np
from models.facenet_model import FaceNetModel


class FaceRecognizer:
    """Recognize faces using normalized FaceNet embeddings."""

    def __init__(self):
        self.facenet = FaceNetModel()
        self.students = []
        self.embeddings = None
        self.load_students()

    def load_students(self):
        base_folder = os.path.join("data", "students")
        if not os.path.exists(base_folder):
            return

        for roll in os.listdir(base_folder):
            folder = os.path.join(base_folder, roll)
            info_path = os.path.join(folder, "info.json")
            embedding_path = os.path.join(folder, "embedding.pkl")

            if not (os.path.exists(info_path) and os.path.exists(embedding_path)):
                continue

            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                with open(embedding_path, "rb") as f:
                    embedding = np.asarray(pickle.load(f), dtype=np.float32)

                norm = np.linalg.norm(embedding)
                if norm == 0:
                    continue
                embedding /= norm

                self.students.append({"info": info, "embedding": embedding})
            except (OSError, ValueError, pickle.PickleError):
                continue

        if self.students:
            self.embeddings = np.vstack([s["embedding"] for s in self.students])

    def recognize(self, face, threshold=0.70):
        if self.embeddings is None or len(self.students) == 0:
            return None, -1.0

        embedding = self.facenet.get_embedding(face).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None, -1.0
        embedding /= norm

        # All vectors are normalized, so cosine similarity = dot product.
        scores = self.embeddings @ embedding
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])

        if best_score >= threshold:
            return self.students[best_index]["info"], best_score

        return None, best_score
