import cv2
import os
import time

from models.face_detector import FaceDetector
import config


class CameraCapture:
    """Capture registration images with lightweight YuNet processing."""

    def __init__(self):
        self.detector = FaceDetector()
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            # Fallback for systems where CAP_DSHOW is unavailable.
            self.camera.release()
            self.camera = cv2.VideoCapture(config.CAMERA_INDEX)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def capture_student(self, images_folder):
        instructions = [
            ("Look Straight", 4),
            ("Look Left", 4),
            ("Look Right", 4),
            ("Look Up", 3),
            ("Look Down", 3),
            ("Smile", 2),
        ]

        image_count = 1
        last_capture = 0.0

        try:
            for instruction, total in instructions:
                captured = 0

                while captured < total:
                    ret, frame = self.camera.read()
                    if not ret:
                        return False

                    faces = self.detector.detect(
                        frame,
                        max_width=config.DETECTION_MAX_WIDTH,
                    )

                    display = frame.copy()
                    cv2.putText(
                        display,
                        f"{instruction}  ({captured + 1}/{total})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                    if faces is not None and len(faces) > 0:
                        frame_h, frame_w = frame.shape[:2]
                        face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = face[:4].astype(int)

                        if w >= 120 and h >= 120 and not (
                            x <= 5 or y <= 5 or
                            x + w >= frame_w - 5 or
                            y + h >= frame_h - 5
                        ):
                            cv2.rectangle(
                                display,
                                (x, y),
                                (x + w, y + h),
                                (0, 255, 0),
                                2,
                            )

                            now = time.monotonic()
                            if now - last_capture >= config.REGISTRATION_CAPTURE_INTERVAL:
                                face_crop = frame[y:y + h, x:x + w]
                                if face_crop.size > 0:
                                    filename = os.path.join(
                                        images_folder,
                                        f"img_{image_count:03}.jpg",
                                    )
                                    # JPEG writing is small because the crop is only a face.
                                    if cv2.imwrite(filename, face_crop):
                                        image_count += 1
                                        captured += 1
                                        last_capture = now

                    cv2.imshow("Student Registration - Press Q to cancel", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return False

            return True
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
