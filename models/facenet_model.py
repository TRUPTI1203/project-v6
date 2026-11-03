import io
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
from keras_facenet import FaceNet


class FaceNetModel:
    """Lazy-loaded FaceNet wrapper compatible with keras-facenet.

    keras-facenet's FaceNet.embeddings() does not accept Keras' ``verbose``
    keyword in all versions.  We therefore call it without that argument.
    Its internal predict() can print a 1/1 progress line, so stdout/stderr
    are temporarily captured during inference to keep the application console
    clean without passing an unsupported keyword.
    """

    def __init__(self):
        self.model = None

    def _load(self):
        if self.model is None:
            self.model = FaceNet()

    @staticmethod
    def _silent_embeddings(model, faces):
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            return model.embeddings(faces)

    def get_embedding(self, face):
        self._load()
        face = np.asarray(face, dtype=np.uint8)
        embeddings = self._silent_embeddings(self.model, [face])
        return embeddings[0]

    def get_embeddings(self, faces):
        self._load()
        faces = [np.asarray(face, dtype=np.uint8) for face in faces]
        return self._silent_embeddings(self.model, faces)
