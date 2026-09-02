import cv2

from ultralytics import YOLO

from app.services.constants import VEHICLE_CLASSES


class VideoProcessor:

    VEHICLE_CLASSES = VEHICLE_CLASSES

    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def process(
        self,
        input_path: str,
        output_path: str,
        vehicle_type: str = "all"
    ):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {input_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        frame_number = 0
        unique_vehicle_ids = set()

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Run YOLO + ByteTrack
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[2, 3, 5, 7],
                verbose=False
            )

            result = results[0]

            if result.boxes is not None:

                boxes = result.boxes

                for i in range(len(boxes)):

                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])

                    if class_id not in self.VEHICLE_CLASSES:
                        continue

                    detected_type = self.VEHICLE_CLASSES[class_id]

                    # Apply vehicle filter
                    if (
                        vehicle_type != "all"
                        and detected_type != vehicle_type
                    ):
                        continue

                    # Bounding box
                    x1, y1, x2, y2 = map(
                        int,
                        boxes.xyxy[i]
                    )

                    # Tracking ID
                    track_id = None

                    if boxes.id is not None:
                        track_id = int(boxes.id[i])
                        unique_vehicle_ids.add(track_id)

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Label
                    if track_id is not None:
                        label = (
                            f"ID {track_id} | "
                            f"{detected_type} "
                            f"{confidence:.2f}"
                        )
                    else:
                        label = (
                            f"{detected_type} "
                            f"{confidence:.2f}"
                        )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            writer.write(frame)

        cap.release()
        writer.release()

        return {
            "frames_processed": frame_number,
            "unique_vehicles": len(unique_vehicle_ids),
            "output_path": output_path
        }