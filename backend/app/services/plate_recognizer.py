import cv2
import easyocr

from ultralytics import YOLO

# Generic international model, plus a version fine-tuned on Nepali plates.
# The two catch different real plates in testing (the fine-tune doesn't
# strictly dominate the original), so both run and results are merged
# rather than picking one.
DEFAULT_MODEL_PATHS = (
    "license_plate_detector.pt",
    "license_plate_detector_nepal.pt"
)

# Two boxes from different models are treated as the same plate (keep only
# the higher-confidence one) once their overlap crosses this IoU threshold
NMS_IOU_THRESHOLD = 0.5


class PlateRecognizer:

    def __init__(
        self,
        model_paths=DEFAULT_MODEL_PATHS,
        languages=("ne", "en")
    ):
        self.models = [YOLO(path) for path in model_paths]
        # 'ne' reads Devanagari-script plates, 'en' reads the newer
        # embossed Latin-character plates used on private vehicles in Nepal
        self.reader = easyocr.Reader(list(languages), gpu=False)

    def read_plates(self, image):
        candidates = []

        for model in self.models:
            results = model(image, verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    candidates.append({
                        "confidence": float(box.conf[0]),
                        "box": (x1, y1, x2, y2)
                    })

        plates = []

        for candidate in self._merge_overlapping(candidates):
            x1, y1, x2, y2 = candidate["box"]
            plate_crop = image[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            text, ocr_confidence = self._read_text(plate_crop)

            plates.append({
                "text": text,
                "detection_confidence": round(candidate["confidence"], 4),
                "ocr_confidence": ocr_confidence,
                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

        return plates

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        if intersection == 0:
            return 0.0

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        return intersection / (area_a + area_b - intersection)

    def _merge_overlapping(self, candidates):
        # Greedy NMS across both models' detections: keep the highest-
        # confidence box in each overlapping cluster, drop the rest
        remaining = sorted(
            candidates, key=lambda c: c["confidence"], reverse=True
        )
        kept = []

        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [
                c for c in remaining
                if self._iou(best["box"], c["box"]) < NMS_IOU_THRESHOLD
            ]

        return kept

    def _read_text(self, plate_crop):
        # Plates are often small in wide traffic shots; upscale and boost
        # contrast so EasyOCR has enough resolved detail to work with
        upscaled = cv2.resize(
            plate_crop, None, fx=8, fy=8, interpolation=cv2.INTER_LANCZOS4
        )
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(
            clipLimit=3.0, tileGridSize=(8, 8)
        ).apply(gray)

        readings = self.reader.readtext(enhanced)

        if not readings:
            return None, None

        text = " ".join(reading[1] for reading in readings).strip()
        confidence = round(min(reading[2] for reading in readings), 4)

        return text, confidence
