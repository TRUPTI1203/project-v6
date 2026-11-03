"""Evaluate the Face Recognition Attendance System.

This module evaluates the recognition pipeline using the images stored for each
registered student. By default it uses leave-one-out evaluation: each image is
used as a test sample while the student's reference embedding is built from the
other images. This avoids comparing an image directly with an enrollment
embedding that was generated from the same image.

Usage (from the project root):
    python -m evaluation.evaluate
    python -m evaluation.evaluate --threshold 0.70
    python -m evaluation.evaluate --test-dir evaluation/test_data
    python -m evaluation.evaluate --camera-seconds 20

For a stronger report result, create an independent test set:
    evaluation/test_data/<roll>/*.jpg
The test images should not be the same images used to create the stored
student embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EvaluationError(RuntimeError):
    pass


def load_student_info(student_root: Path):
    students = {}
    if not student_root.is_dir():
        return students

    for folder in sorted(student_root.iterdir()):
        if not folder.is_dir():
            continue
        info_path = folder / "info.json"
        embedding_path = folder / "embedding.pkl"
        if not info_path.exists() or not embedding_path.exists():
            continue
        try:
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        students[folder.name] = info
    return students


def image_files(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_face_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


def cosine_matrix(a, b):
    """Return cosine similarities for normalized rows in a and b."""
    return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32).T


def macro_metrics(actual, predicted, labels):
    """Compute accuracy, macro precision, recall and F1 without extra packages."""
    if not actual:
        return {"accuracy": 0.0, "precision_macro": 0.0,
                "recall_macro": 0.0, "f1_macro": 0.0}

    correct = sum(a == p for a, p in zip(actual, predicted))
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": correct / len(actual),
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }


def evaluate_recognition(project_root: Path, threshold: float, test_dir: Path | None):
    students_root = project_root / "data" / "students"
    info = load_student_info(students_root)
    if not info:
        raise EvaluationError("No registered students with info.json/embedding.pkl were found.")

    # Independent test data, if supplied, has priority. Otherwise use the
    # registration images with leave-one-out references.
    independent = test_dir is not None and test_dir.exists()
    source_root = test_dir if independent else students_root

    samples = []
    per_student_images = {}
    for roll in sorted(info):
        folder = source_root / roll
        if independent:
            files = image_files(folder)
        else:
            files = image_files(folder / "images")
        if files:
            per_student_images[roll] = files
            for path in files:
                samples.append((roll, path))

    if not samples:
        raise EvaluationError(
            "No test images found. Put independent images under "
            "evaluation/test_data/<roll>/ or keep registration images under "
            "data/students/<roll>/images/."
        )

    from models.facenet_model import FaceNetModel
    model = FaceNetModel()

    # Batch embedding generation is used for speed and consistent preprocessing.
    valid_samples = []
    faces = []
    for roll, path in samples:
        face = load_face_image(path)
        if face is not None:
            valid_samples.append((roll, path))
            faces.append(face)

    if not faces:
        raise EvaluationError("All test images failed to load.")

    t0 = time.perf_counter()
    embeddings = model.get_embeddings(faces)
    batch_seconds = time.perf_counter() - t0
    embeddings = np.asarray([normalize(e) for e in embeddings], dtype=np.float32)
    if np.any(~np.isfinite(embeddings)):
        raise EvaluationError("At least one test embedding could not be normalized.")

    # Build reference embeddings.
    references = {}
    stored_embeddings = {}
    for roll in info:
        p = students_root / roll / "embedding.pkl"
        try:
            import pickle
            with p.open("rb") as f:
                stored = normalize(pickle.load(f))
            if stored is not None:
                stored_embeddings[roll] = stored
        except (OSError, ValueError, EOFError, ImportError, pickle.PickleError):
            pass

    if independent:
        # For independent test images, use the project's actual stored
        # enrollment embeddings.
        references = stored_embeddings
    else:
        # Leave-one-out reference for every student's test image.
        by_student = defaultdict(list)
        for (roll, _), emb in zip(valid_samples, embeddings):
            by_student[roll].append(emb)
        for roll, embs in by_student.items():
            references[roll] = np.mean(embs, axis=0)
            references[roll] = normalize(references[roll])

    reference_rolls = sorted(references)
    reference_matrix = np.vstack([references[r] for r in reference_rolls])

    actual = []
    predicted = []
    best_scores = []
    genuine_scores = []
    impostor_scores = []
    rows = []

    # Recognition-level evaluation. For leave-one-out, the true class reference
    # is recomputed for each test image so the test image is not in its reference.
    for idx, ((roll, path), emb) in enumerate(zip(valid_samples, embeddings)):
        if independent:
            matrix = reference_matrix
            rolls = reference_rolls
        else:
            # Build references excluding the current sample.
            dynamic = {}
            for r in reference_rolls:
                student_embs = [e for (rr, _), e in zip(valid_samples, embeddings) if rr == r]
                if r == roll:
                    student_embs = [e for j, ((rr, _), e) in enumerate(zip(valid_samples, embeddings))
                                    if rr == r and j != idx]
                if student_embs:
                    dynamic[r] = normalize(np.mean(student_embs, axis=0))
            rolls = sorted(dynamic)
            matrix = np.vstack([dynamic[r] for r in rolls])

        scores = matrix @ emb
        best_idx = int(np.argmax(scores))
        best_roll = rolls[best_idx]
        best_score = float(scores[best_idx])
        pred = best_roll if best_score >= threshold else "UNKNOWN"

        actual.append(roll)
        predicted.append(pred)
        best_scores.append(best_score)

        own_index = rolls.index(roll) if roll in rolls else None
        own_score = float(scores[own_index]) if own_index is not None else None
        if own_score is not None:
            genuine_scores.append(own_score)

        for r, score in zip(rolls, scores):
            if r != roll:
                impostor_scores.append(float(score))

        rows.append({
            "actual_roll": roll,
            "predicted_roll": pred,
            "similarity": round(best_score, 6),
            "correct": pred == roll,
            "image": str(path.relative_to(project_root)),
        })

    labels = sorted(info)
    metrics = macro_metrics(actual, predicted, labels)

    # Threshold-based verification metrics.
    far = (sum(s >= threshold for s in impostor_scores) / len(impostor_scores)
           if impostor_scores else 0.0)
    frr = (sum(s < threshold for s in genuine_scores) / len(genuine_scores)
           if genuine_scores else 0.0)

    confusion = Counter((a, p) for a, p in zip(actual, predicted))

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "independent_test_set" if independent else "leave_one_out_registration_images",
        "warning": (
            None if independent else
            "These results use registration images with leave-one-out references. "
            "They are useful for a baseline but are not an independent test-set result."
        ),
        "threshold": threshold,
        "registered_students": len(info),
        "test_samples": len(valid_samples),
        "skipped_images": len(samples) - len(valid_samples),
        **metrics,
        "false_acceptance_rate": far,
        "false_rejection_rate": frr,
        "average_batch_embedding_time_ms": (batch_seconds * 1000 / len(faces)),
        "genuine_similarity_mean": float(np.mean(genuine_scores)) if genuine_scores else 0.0,
        "genuine_similarity_min": float(np.min(genuine_scores)) if genuine_scores else 0.0,
        "impostor_similarity_max": float(np.max(impostor_scores)) if impostor_scores else 0.0,
    }

    return result, rows, confusion


def benchmark_facenet(project_root: Path, samples: int = 20):
    """Measure warm FaceNet single-image inference latency."""
    student_root = project_root / "data" / "students"
    candidates = []
    for roll in sorted(os.listdir(student_root)) if student_root.exists() else []:
        candidates.extend(image_files(student_root / roll / "images"))
    if not candidates:
        return {}

    face = load_face_image(candidates[0])
    if face is None:
        return {}

    from models.facenet_model import FaceNetModel
    model = FaceNetModel()
    model.get_embedding(face)  # warm-up
    times = []
    for _ in range(max(1, samples)):
        start = time.perf_counter()
        model.get_embedding(face)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "samples": len(times),
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def benchmark_camera(seconds: int):
    """Measure webcam + YuNet + FaceNet recognition without writing attendance.

    The benchmark intentionally does not use AttendanceManager, so running it
    cannot add rows to the real attendance CSV. Recognition is performed at the
    same interval configured by the application.
    """
    if seconds <= 0:
        return {}

    from models.face_detector import FaceDetector
    from utils.recognition import FaceRecognizer

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(config.CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        raise EvaluationError("Camera could not be opened for the FPS benchmark.")

    frames = 0
    detections = 0
    recognitions = 0
    recognized = 0
    unknown = 0
    recognition_times = []
    start = time.perf_counter()
    next_recognition = 0.0

    try:
        while time.perf_counter() - start < seconds:
            ok, frame = camera.read()
            if not ok:
                continue
            frames += 1
            faces = detector.detect(frame, max_width=config.DETECTION_MAX_WIDTH)
            if faces is None or len(faces) == 0:
                continue

            detections += len(faces)
            now = time.monotonic()
            if now < next_recognition:
                continue

            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face[:4].astype(int)
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            if w <= 0 or h <= 0:
                continue

            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)

            t0 = time.perf_counter()
            info, _score = recognizer.recognize(
                crop, threshold=config.SIMILARITY_THRESHOLD
            )
            recognition_times.append((time.perf_counter() - t0) * 1000)
            recognitions += 1
            if info is None:
                unknown += 1
            else:
                recognized += 1
            next_recognition = time.monotonic() + config.RECOGNITION_INTERVAL
    finally:
        camera.release()

    elapsed = time.perf_counter() - start
    return {
        "seconds": round(elapsed, 3),
        "frames": frames,
        "camera_fps": frames / elapsed if elapsed else 0.0,
        "detected_faces": detections,
        "recognitions": recognitions,
        "recognitions_per_second": recognitions / elapsed if elapsed else 0.0,
        "recognized": recognized,
        "unknown": unknown,
        "recognition_latency_mean_ms": float(np.mean(recognition_times)) if recognition_times else 0.0,
        "recognition_latency_median_ms": float(np.median(recognition_times)) if recognition_times else 0.0,
        "recognition_latency_min_ms": float(np.min(recognition_times)) if recognition_times else 0.0,
        "recognition_latency_max_ms": float(np.max(recognition_times)) if recognition_times else 0.0,
    }

def write_outputs(project_root: Path, result, rows, confusion, facenet, camera):
    out = project_root / "evaluation" / "results"
    out.mkdir(parents=True, exist_ok=True)

    result = dict(result)
    result["facenet_latency"] = facenet
    result["camera_benchmark"] = camera

    with (out / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with (out / "sample_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["actual_roll", "predicted_roll", "similarity", "correct", "image"])
        writer.writeheader()
        writer.writerows(rows)

    labels = sorted(set(result_label for pair in confusion for result_label in pair))
    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual \\ Predicted"] + labels)
        for actual in labels:
            writer.writerow([actual] + [confusion.get((actual, predicted), 0) for predicted in labels])

    return out


def print_summary(result, facenet, camera):
    print("\n=== Face Recognition Performance Evaluation ===")
    print(f"Mode                    : {result['evaluation_mode']}")
    if result.get("warning"):
        print(f"NOTE                    : {result['warning']}")
    print(f"Registered students     : {result['registered_students']}")
    print(f"Test samples            : {result['test_samples']}")
    print(f"Threshold               : {result['threshold']:.3f}")
    print(f"Accuracy                : {result['accuracy'] * 100:.2f}%")
    print(f"Macro precision         : {result['precision_macro'] * 100:.2f}%")
    print(f"Macro recall            : {result['recall_macro'] * 100:.2f}%")
    print(f"Macro F1-score          : {result['f1_macro'] * 100:.2f}%")
    print(f"False acceptance rate   : {result['false_acceptance_rate'] * 100:.2f}%")
    print(f"False rejection rate    : {result['false_rejection_rate'] * 100:.2f}%")
    print(f"Mean genuine similarity : {result['genuine_similarity_mean']:.4f}")
    print(f"Max impostor similarity : {result['impostor_similarity_max']:.4f}")
    print(f"Batch embedding / face  : {result['average_batch_embedding_time_ms']:.2f} ms")

    if facenet:
        print("\n--- FaceNet latency (warm single-image inference) ---")
        print(f"Samples                 : {facenet['samples']}")
        print(f"Mean                    : {facenet['mean_ms']:.2f} ms")
        print(f"Median                  : {facenet['median_ms']:.2f} ms")
        print(f"Min / Max               : {facenet['min_ms']:.2f} / {facenet['max_ms']:.2f} ms")

    if camera:
        print("\n--- Camera benchmark ---")
        print(f"Measured FPS            : {camera['camera_fps']:.2f}")
        print(f"Frames                  : {camera['frames']}")
        print(f"Detected faces          : {camera['detected_faces']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate face-recognition performance.")
    parser.add_argument("--threshold", type=float, default=config.SIMILARITY_THRESHOLD,
                        help="Recognition threshold (default: config value).")
    parser.add_argument("--test-dir", type=Path, default=None,
                        help="Independent test root: <test-dir>/<roll>/*.jpg")
    parser.add_argument("--latency-samples", type=int, default=20,
                        help="Warm FaceNet latency measurements.")
    parser.add_argument("--camera-seconds", type=int, default=0,
                        help="Also benchmark webcam + YuNet for N seconds (0 disables).")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project root directory.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    test_dir = args.test_dir
    if test_dir is not None and not test_dir.is_absolute():
        test_dir = project_root / test_dir

    result, rows, confusion = evaluate_recognition(project_root, args.threshold, test_dir)
    facenet = benchmark_facenet(project_root, args.latency_samples)
    camera = benchmark_camera(args.camera_seconds)
    out = write_outputs(project_root, result, rows, confusion, facenet, camera)

    print_summary(result, facenet, camera)
    print(f"\nSaved evaluation files to: {out}")


if __name__ == "__main__":
    main()
