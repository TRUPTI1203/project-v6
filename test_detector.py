import cv2

from models.face_detector import FaceDetector


camera = cv2.VideoCapture(0)

detector = FaceDetector()


while True:

    ret, frame = camera.read()

    if not ret:
        break

    faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(

                frame,

                (x, y),

                (x + w, y + h),

                (0, 255, 0),

                2

            )

    cv2.imshow("Face Detector", frame)

    if cv2.waitKey(1) == ord("q"):
        break


camera.release()

cv2.destroyAllWindows()