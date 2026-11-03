import cv2
import time
import threading
from queue import Queue, Full

import config
from utils.recognition import FaceRecognizer
from utils.attendance_manager import AttendanceManager
from models.face_detector import FaceDetector


class RecognizeController:
    """Run camera/AI work in a background thread so Tkinter stays responsive."""

    def __init__(self, page):
        self.page = page
        self.camera = None
        self.running = False
        self.worker = None

        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=5)
        self.last_recognition_time = 0.0

    def start(self):
        if self.running:
            return

        self.running = True
        self.last_recognition_time = 0.0
        self.worker = threading.Thread(
            target=self._camera_worker,
            name="AttendanceCameraWorker",
            daemon=True,
        )
        self.worker.start()

    def stop(self):
        """Stop the worker and wait briefly so the webcam is fully released.

        This is important when navigating Attendance -> Register. Without waiting
        for the worker to finish, the registration camera can start while the
        attendance camera still owns the webcam.
        """
        self.running = False

        camera = self.camera
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass

        worker = self.worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)

        self.worker = None
        self.camera = None

    def _put_latest_frame(self, frame):
        try:
            self.frame_queue.get_nowait()
        except Exception:
            pass
        try:
            self.frame_queue.put_nowait(frame)
        except Full:
            pass

    def _put_result(self, info, confidence, status):
        try:
            self.result_queue.put_nowait((info, confidence, status))
        except Full:
            try:
                self.result_queue.get_nowait()
            except Exception:
                pass
            try:
                self.result_queue.put_nowait((info, confidence, status))
            except Full:
                pass

    def _camera_worker(self):
        camera = None
        try:
            camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(config.CAMERA_INDEX)

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera = camera

            if not camera.isOpened():
                self._put_result(None, -1.0, "Camera could not be opened")
                return

            # If the user navigated away while the camera was opening, stop now.
            if not self.running:
                return

            # Load the AI components only after the camera is ready. They are kept
            # in this worker thread so model initialization never freezes Tkinter.
            detector = FaceDetector()
            recognizer = FaceRecognizer()
            attendance = AttendanceManager()

            if not self.running:
                return

            frame_number = 0
            last_faces = None
            next_recognition = 0.0

            while self.running:
                ret, frame = camera.read()
                if not ret:
                    break

                frame_number += 1

                if frame_number % 2 == 0 or last_faces is None:
                    last_faces = detector.detect(
                        frame,
                        max_width=config.DETECTION_MAX_WIDTH,
                    )

                if last_faces is not None and len(last_faces) > 0:
                    face = max(last_faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = face[:4].astype(int)

                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2,
                    )

                    now = time.monotonic()
                    if now >= next_recognition and w > 0 and h > 0:
                        face_crop = frame[y:y + h, x:x + w]
                        if face_crop.size > 0:
                            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            face_crop = cv2.resize(
                                face_crop,
                                (160, 160),
                                interpolation=cv2.INTER_AREA,
                            )

                            info, confidence = recognizer.recognize(
                                face_crop,
                                threshold=config.SIMILARITY_THRESHOLD,
                            )

                            if info is not None:
                                marked = attendance.mark(info)
                                status = "Attendance Marked" if marked else "Already Marked"
                                self._put_result(info, confidence, status)
                            else:
                                self._put_result(None, confidence, "Unknown Face")

                            next_recognition = now + config.RECOGNITION_INTERVAL

                self._put_latest_frame(frame)

        except Exception as exc:
            self._put_result(None, -1.0, f"Camera error: {exc}")
        finally:
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    pass
            self.camera = None
            self.running = False

    def poll_ui(self):
        """Called by Tkinter periodically to display worker results."""
        if not self.page.winfo_exists():
            return

        latest_frame = None
        while True:
            try:
                latest_frame = self.frame_queue.get_nowait()
            except Exception:
                break

        if latest_frame is not None:
            self.page.show_camera_frame(latest_frame)

        while True:
            try:
                info, confidence, status = self.result_queue.get_nowait()
            except Exception:
                break

            self.page.update_student(info, confidence, status)

        return
