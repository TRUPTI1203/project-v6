import cv2


class FaceDetector:
    """Fast YuNet wrapper.

    The detector works on a resized frame. The returned bounding boxes are
    mapped back to the original frame coordinates.
    """

    def __init__(self, model_path="models/face_detection_yunet_2023mar.onnx",
                 input_size=(320, 320), score_threshold=0.8,
                 nms_threshold=0.3, top_k=5000):
        self.input_size = input_size
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
        self._last_input_size = input_size

    def detect(self, frame, max_width=640):
        h, w = frame.shape[:2]
        scale = min(1.0, max_width / float(w))

        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            small = frame
            new_w, new_h = w, h
            scale = 1.0

        size = (new_w, new_h)
        if size != self._last_input_size:
            self.detector.setInputSize(size)
            self._last_input_size = size

        _, faces = self.detector.detect(small)

        if faces is None:
            return None

        if scale != 1.0:
            faces = faces.copy()
            faces[:, :4] /= scale
            # YuNet also returns 5 landmark pairs after x,y,w,h.
            if faces.shape[1] > 4:
                faces[:, 4:] /= scale

        return faces
