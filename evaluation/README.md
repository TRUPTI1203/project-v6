# Performance / Evaluation Module

This module measures the performance of the actual Face Recognition Attendance System.

## 1. Baseline evaluation

From the project root:

```bash
python -m evaluation.evaluate
```

If no independent test set is supplied, the module uses **leave-one-out evaluation** on the registration images. Each image is tested against a reference embedding built from the other images of the same student. This is better than testing an image directly against an embedding that contains that same image, but it is still **not an independent test set**.

The command produces:

```text
evaluation/results/evaluation_summary.json
evaluation/results/sample_results.csv
evaluation/results/confusion_matrix.csv
```

## 2. Independent evaluation (recommended for the report)

Create this structure:

```text
evaluation/
└── test_data/
    ├── 23/
    │   ├── test_01.jpg
    │   └── test_02.jpg
    ├── 24/
    │   ├── test_01.jpg
    │   └── test_02.jpg
    └── ...
```

The folder names must match the registered student roll numbers. The test images should be newly captured images and should **not** be the same images used during registration.

Run:

```bash
python -m evaluation.evaluate --test-dir evaluation/test_data
```

This uses the project's stored `embedding.pkl` files as the enrollment/reference embeddings.

## 3. Threshold experiments

To compare thresholds:

```bash
python -m evaluation.evaluate --threshold 0.75
python -m evaluation.evaluate --threshold 0.80
python -m evaluation.evaluate --threshold 0.85
```

Save the generated summaries separately if you want to compare FAR/FRR and recognition accuracy.

## 4. Measure camera performance

To benchmark webcam capture + YuNet processing on the machine running the project:

```bash
python -m evaluation.evaluate --camera-seconds 20
```

This measures camera FPS, YuNet processing, and the actual FaceNet recognition latency without writing anything to the attendance CSV. It is safe to use for runtime benchmarking.

## 5. What to report

Use the measured values for:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Mean genuine similarity
- Maximum impostor similarity
- FaceNet inference latency
- Camera FPS

Do not use the numbers from a registration-image baseline as if they were an independent test-set result.
