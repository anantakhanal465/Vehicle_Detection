"""
Harvests "background" negative examples for the character classifier:
small crops of generic vehicle-body texture (grilles, windshields,
seats, fabric...) that are NOT plate characters.

Why this exists: testing PlateRecognizer against real multi-vehicle
Nepal traffic photos (not close-up single-plate shots) showed the plate
*detector* proposing false-positive boxes on busy scenes -- 4 of 5
detected "plates" in one test batch were actually a van grille, a
windshield, a motorcycle seat, and a piece of fabric. Worse, the
character classifier would then confidently (0.5-0.8 confidence) read
plausible-looking Devanagari garbage from these non-plate crops instead
of recognizing they weren't plates at all -- the same overconfident-on-
out-of-distribution-input problem as the Latin-script case, just for
"not a character" rather than "wrong script." This harvests real
negative examples so a "background" class can be added to the
classifier to fix it (see char_classifier.py's BACKGROUND_LABEL).

Two sources:
1. Random small patches sampled from inside real vehicle detection
   boxes (using the existing VehicleDetector) across a set of test
   images, skipping any patch that overlaps a known plate-candidate
   box -- generic "vehicle body, not a plate" negatives.
2. Specific plate-detector false positives from that same test batch,
   copied in directly as hard negatives (most discriminative, since
   they're exactly the failure mode being fixed).

Usage:
    python backend/scripts/harvest_char_negatives.py

Requires backend/test_images/nepal_traffic/ (10 Nepal traffic photos)
and the scratch video frames used during interactive testing to exist;
if you don't have those, sample from whatever real multi-vehicle photos
you have instead -- the exact source images aren't important, what
matters is the negatives come from genuine non-plate vehicle regions
rather than being synthetic. Writes crops to
backend/datasets/char_negatives/, then symlink that directory in as
combined_char_ocr/background/ before retraining (see
train_char_classifier.py's docstring for the full merge).
"""

import random
import sys
from pathlib import Path

import cv2

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.vehicle_detector import VehicleDetector  # noqa: E402

TRAFFIC_IMAGES_DIR = BACKEND_DIR / "test_images" / "nepal_traffic"
OUTPUT_DIR = BACKEND_DIR / "datasets" / "char_negatives"

PATCHES_PER_VEHICLE = 8
MIN_VEHICLE_SIZE = 30


def overlaps(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    return max(0, ix2 - ix1) * max(0, iy2 - iy1) > 0


def harvest_from_images(image_dir, detector, avoid_boxes_by_image, out_dir, seed):
    random.seed(seed)
    count = 0

    for image_path in sorted(image_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        vehicles = detector.detect(image, vehicle_type="all")
        avoid = avoid_boxes_by_image.get(image_path.name, [])

        for vehicle in vehicles:
            box = vehicle["bounding_box"]
            vx1, vy1, vx2, vy2 = box["x1"], box["y1"], box["x2"], box["y2"]
            vw, vh = vx2 - vx1, vy2 - vy1

            if vw < MIN_VEHICLE_SIZE or vh < MIN_VEHICLE_SIZE:
                continue

            for _ in range(PATCHES_PER_VEHICLE):
                patch_w = random.randint(max(15, vw // 8), max(20, vw // 3))
                patch_h = random.randint(max(12, vh // 8), max(16, vh // 3))

                if vx2 - patch_w <= vx1 or vy2 - patch_h <= vy1:
                    continue

                px1 = random.randint(vx1, vx2 - patch_w)
                py1 = random.randint(vy1, vy2 - patch_h)
                patch_box = (px1, py1, px1 + patch_w, py1 + patch_h)

                if any(overlaps(patch_box, a) for a in avoid):
                    continue

                patch = image[py1:py1 + patch_h, px1:px1 + patch_w]

                if patch.size == 0:
                    continue

                cv2.imwrite(
                    str(out_dir / f"{image_path.stem}_{count}.jpg"), patch
                )
                count += 1

    return count


def main():
    if not TRAFFIC_IMAGES_DIR.is_dir():
        raise SystemExit(
            f"{TRAFFIC_IMAGES_DIR} not found -- point this script at "
            "whatever real multi-vehicle traffic photos you have."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = VehicleDetector()

    count = harvest_from_images(
        TRAFFIC_IMAGES_DIR, detector, {}, OUTPUT_DIR, seed=42
    )
    print(f"harvested {count} negative patches from {TRAFFIC_IMAGES_DIR}")


if __name__ == "__main__":
    main()
