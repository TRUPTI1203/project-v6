import os
import cv2
import pickle
import numpy as np
from models.facenet_model import FaceNetModel


class EmbeddingGenerator:
    """Generate one representative embedding using one batched FaceNet call."""

    def __init__(self):
        self.facenet = FaceNetModel()

    def generate_student_embedding(self, images_folder, output_file):
        faces = []

        for image_name in sorted(os.listdir(images_folder)):
            image_path = os.path.join(images_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
            faces.append(image)

        if not faces:
            raise ValueError("No valid face images were captured.")

        # One batched inference is much faster than calling FaceNet 20 times.
        embeddings = self.facenet.get_embeddings(faces)
        average_embedding = np.mean(embeddings, axis=0)

        norm = np.linalg.norm(average_embedding)
        if norm == 0:
            raise ValueError("Generated embedding has zero magnitude.")

        average_embedding = average_embedding / norm

        with open(output_file, "wb") as file:
            pickle.dump(average_embedding.astype(np.float32), file)

        return average_embedding
