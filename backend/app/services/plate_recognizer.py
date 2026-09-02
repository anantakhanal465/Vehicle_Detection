import cv2
import easyocr

from ultralytics import YOLO


class PlateRecognizer:

    def __init__(
        self,
        model_path="license_plate_detector.pt",
        languages=("ne", "en")
    ):
        self.model = YOLO(model_path)
        # 'ne' reads Devanagari-script plates, 'en' reads the newer
        # embossed Latin-character plates used on private vehicles in Nepal
        self.reader = easyocr.Reader(list(languages), gpu=False)

    def read_plates(self, image):
        results = self.model(image, verbose=False)

        plates = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate_crop = image[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                text, ocr_confidence = self._read_text(plate_crop)

                plates.append({
                    "text": text,
                    "detection_confidence": round(confidence, 4),
                    "ocr_confidence": ocr_confidence,
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

        return plates

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
