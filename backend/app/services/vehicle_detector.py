import threading

from ultralytics import YOLO

from app.services.constants import VEHICLE_CLASSES


class VehicleDetector:

    VEHICLE_CLASSES = VEHICLE_CLASSES

    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        # This instance is shared across requests; Ultralytics YOLO models
        # aren't guaranteed safe for concurrent inference from multiple
        # threads (e.g. a threadpooled video request overlapping a
        # synchronous image request), so serialize access to it.
        self._lock = threading.Lock()

    def detect(self, image, vehicle_type="all"):

        with self._lock:
            results = self.model(image)

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id not in self.VEHICLE_CLASSES:
                    continue

                detected_type = self.VEHICLE_CLASSES[class_id]

                # Apply frontend-selected vehicle filter
                if vehicle_type != "all" and detected_type != vehicle_type:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "vehicle_type": detected_type,
                    "confidence": round(confidence, 4),
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

        return detections 