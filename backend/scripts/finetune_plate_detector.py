"""Fine-tune the license plate detector on Nepali vehicle photos.

Produces backend/license_plate_detector_nepal.pt, which is used alongside
the original generic detector as an ensemble in PlateRecognizer (see
app/services/plate_recognizer.py) — not as a replacement. Testing against
real traffic footage (backend/videos/nepal_test.mp4) showed the two models
catch different real plates rather than one strictly dominating the other,
so both run and results are merged.

Background — two things were tried before this:

1. A full fine-tune (20 epochs, default LR, no frozen layers) reached
   mAP50 0.98 on its own held-out validation split, but catastrophically
   overfit to the training dataset's domain (close-range phone photos,
   plate filling 20-40% of frame, dominated by red Nepali plates) and
   fixated on unrelated red storefront signage as a false positive on
   every single frame of real aerial traffic footage. The validation
   metric was misleading because the validation split came from the same
   narrow domain as training.

2. This script — a much gentler fine-tune (3 epochs, low LR, frozen
   backbone) — avoided that regression while still learning something
   useful, verified by testing against the same real footage.

Dataset: inspiring-lab/nepali-vehicles-number-plate-dataset (Kaggle,
Apache-2.0), ~8,078 images with single-class bounding boxes around
license plates. Downloaded via the Kaggle CLI (`kaggle datasets download
-d ishworsubedii/vehicle-number-plate-datasetnepal`) into
backend/datasets/vehicle-number-plate-datasetnepal/, then split 90/10
into backend/datasets/nepal_plate_yolo/.

Run from backend/:
    python3 scripts/finetune_plate_detector.py
"""

from ultralytics import YOLO


def main():
    model = YOLO("license_plate_detector.pt")

    model.train(
        data="datasets/nepal_plate_yolo/data.yaml",
        epochs=3,
        imgsz=320,
        batch=16,
        device="cpu",
        optimizer="AdamW",
        lr0=0.0003,
        # Freeze the backbone (layers 0-9) so the model only adapts its
        # detection head to Nepali plates instead of relearning general
        # features — this is what avoided the overfitting seen with a
        # full unfrozen fine-tune.
        freeze=10,
        project="datasets/finetune_runs",
        name="nepal_plates_gentle",
        patience=0,
        exist_ok=True
    )

    print(
        "\nBest weights: "
        "datasets/finetune_runs/nepal_plates_gentle/weights/best.pt\n"
        "Copy to backend/license_plate_detector_nepal.pt to deploy it."
    )


if __name__ == "__main__":
    main()
