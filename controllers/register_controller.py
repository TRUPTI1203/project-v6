"""Student registration controller."""

import os
import shutil
import threading

from utils.camera import CameraCapture
from utils.helper import create_student_folder, save_student_info
from utils.embedding import EmbeddingGenerator


class RegisterController:
    def __init__(self):
        self.embedding = None

    def validate(self, student):
        if not student.name.strip():
            return False, "Student name cannot be empty."
        if not student.roll.strip():
            return False, "Roll number cannot be empty."
        if not student.branch.strip():
            return False, "Please select a branch."
        if not student.semester.strip():
            return False, "Please select semester."
        if not student.section.strip():
            return False, "Please select section."
        return True, "Validation Successful"

    def student_exists(self, roll):
        return os.path.exists(os.path.join("data", "students", roll))

    def register_student(self, student):
        valid, message = self.validate(student)
        if not valid:
            return False, message

        if self.student_exists(student.roll):
            return False, "Student already registered."

        folder, images_folder = create_student_folder(student.roll)
        save_student_info(student, folder)

        try:
            camera = CameraCapture()
            success = camera.capture_student(images_folder)
            if not success:
                shutil.rmtree(folder, ignore_errors=True)
                return False, "Registration cancelled."

            # Lazy-load FaceNet only after capture is complete.
            if self.embedding is None:
                self.embedding = EmbeddingGenerator()

            self.embedding.generate_student_embedding(
                images_folder,
                os.path.join(folder, "embedding.pkl"),
            )
            return True, "Student registered successfully."

        except Exception as exc:
            shutil.rmtree(folder, ignore_errors=True)
            return False, f"Registration failed: {exc}"

    def register_student_async(self, student, callback):
        """Run camera capture + FaceNet embedding off the Tkinter main thread."""
        def worker():
            result = self.register_student(student)
            callback(*result)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
