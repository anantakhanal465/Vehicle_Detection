import logging
import threading

import cv2
import easyocr

from ultralytics import YOLO

from app.services.char_classifier import CharClassifier

logger = logging.getLogger(__name__)

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

# Below this mean per-character confidence, the character-classifier
# reading isn't trusted and EasyOCR's reading is used instead. Chosen
# from the classifier's own ~0.95+ validation accuracy on isolated
# characters -- a low mean here usually means segmentation split a
# character wrong (e.g. a Latin-embossed plate the classifier was never
# trained on), not that the plate is genuinely hard to read.
CHAR_CLASSIFIER_CONFIDENCE_THRESHOLD = 0.5


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

        # Trained on real Nepali plate character crops (see
        # scripts/train_char_classifier.py) -- reads Devanagari plates far
        # more reliably than EasyOCR's generic 'ne' model, which is trained
        # on printed/handwritten documents rather than the embossed stencil
        # font plates actually use. Optional: falls back to EasyOCR-only if
        # the model weights haven't been trained/placed yet.
        try:
            self.char_classifier = CharClassifier()
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                "Character classifier unavailable (%s); falling back to "
                "EasyOCR only. Run scripts/train_char_classifier.py to "
                "enable it.", exc
            )
            self.char_classifier = None
        # This instance is shared across requests (the synchronous /plate
        # endpoint and threadpooled video processing can both call it at
        # once); neither the YOLO models nor the EasyOCR reader are
        # guaranteed safe for concurrent inference from multiple threads.
        self._lock = threading.Lock()

    def read_plates(self, image):
        with self._lock:
            return self._read_plates_locked(image)

    def _read_plates_locked(self, image):
        candidates = self._detect_candidates(image)

        if not candidates:
            candidates = self._detect_candidates_padded(image)

        plates = []

        for candidate in self._merge_overlapping(candidates):
            x1, y1, x2, y2 = candidate["box"]
            plate_crop = image[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # Detection boxes are sometimes a little tight, especially
            # from the padded-retry path -- cropping exactly to the box
            # can cut off part of a character on the plate's edge (a real
            # crop went from misreading "०२ २७" to the correct "०२३ २७६"
            # once the missing edge pixels were included). Read text from
            # a slightly wider crop than what's reported/drawn as the
            # plate's bounding box.
            ocr_crop = self._expand_crop(image, (x1, y1, x2, y2))
            text, ocr_confidence, is_background = self._read_text(ocr_crop)

            if is_background:
                # The character classifier segmented this candidate and
                # most of what it found looks like generic vehicle
                # texture (grille, windshield, seat, fabric...), not
                # plate characters -- the plate *detector* proposed a
                # box here, but this almost certainly isn't a real
                # plate. Drop it rather than reporting a box with
                # garbage or empty text (verified: on a set of real
                # traffic photos, 4 of 5 detector-proposed "plates" were
                # actually false positives like this).
                continue

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

    def _detect_candidates(self, image):
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

        return candidates

    def _detect_candidates_padded(self, image):
        # Both models were trained on scenes where a plate is a small
        # region surrounded by other context (vehicle body, road) and rely
        # on that surrounding context to localize it. A close-up photo
        # where the plate fills nearly the whole frame has no context
        # pixels, and confidence collapses to near-zero even though the
        # plate itself is large and clear (verified: a real close-up plate
        # crop went from ~0.01 confidence unpadded to ~0.55 padded). Retry
        # once with a padded border before giving up.
        height, width = image.shape[:2]
        pad = int(max(height, width) * 0.5)
        padded = cv2.copyMakeBorder(
            image, pad, pad, pad, pad, cv2.BORDER_REPLICATE
        )

        candidates = self._detect_candidates(padded)

        for candidate in candidates:
            x1, y1, x2, y2 = candidate["box"]
            candidate["box"] = (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(width, x2 - pad),
                min(height, y2 - pad)
            )

        return candidates

    @staticmethod
    def _expand_crop(image, box, margin_frac=0.2):
        x1, y1, x2, y2 = box
        height, width = image.shape[:2]
        margin_x = int((x2 - x1) * margin_frac)
        margin_y = int((y2 - y1) * margin_frac)

        return image[
            max(0, y1 - margin_y):min(height, y2 + margin_y),
            max(0, x1 - margin_x):min(width, x2 + margin_x)
        ]

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
        if self.char_classifier is not None:
            text, confidence, is_background = (
                self.char_classifier.read_plate_text(plate_crop)
            )

            if is_background:
                return None, None, True

            if (
                text is not None
                and confidence >= CHAR_CLASSIFIER_CONFIDENCE_THRESHOLD
            ):
                return text, confidence, False

        text, confidence = self._read_text_easyocr(plate_crop)

        return text, confidence, False

    def _read_text_easyocr(self, plate_crop):
        # Plates are often small in wide traffic shots; upscale and boost
        # contrast so EasyOCR has enough resolved detail to work with.
        #
        # Dropping CLAHE was tried here after it looked like a clear win
        # on one crop (0.068 -> 0.685 confidence), but a broader test
        # across ~16 plate boxes from several frames of the same video
        # showed CLAHE winning more often than not overall — that one
        # crop wasn't representative. Most readings either way are still
        # low-confidence given how small these crops are; CLAHE is kept
        # as the better default on the fuller sample, not reverted blindly.
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
 