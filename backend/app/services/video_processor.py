import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO

from app.services.constants import VEHICLE_CLASSES

# cv2.putText's built-in font can't render non-Latin scripts (it silently
# substitutes '?'), which breaks Devanagari plate-text overlays. Render
# labels through PIL with a bundled font instead.
FONT_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets" / "fonts" / "NotoSansDevanagari-Medium.ttf"
)


class VideoProcessor:

    VEHICLE_CLASSES = VEHICLE_CLASSES

    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self._font = ImageFont.truetype(str(FONT_PATH), 22)

    def process(
        self,
        input_path: str,
        output_path: str,
        vehicle_type: str = "all",
        plate_recognizer=None,
        plate_retry_interval: int = 20,
        plate_confidence_threshold: float = 0.4,
        max_plate_attempts: int = 5
    ):
        # Per track_id: {"text", "confidence", "attempts", "last_attempt_frame"}
        track_plates = {}

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

        # This build of opencv-python can't encode real H.264 (no libx264,
        # and its hardware-encoder fallback has no device to use here), so
        # it writes MPEG-4 Part 2 instead. That plays fine in VLC/ffplay
        # but browsers can't decode it at all — the output gets transcoded
        # to H.264 via ffmpeg below before being handed back as output_path.
        raw_output_path = Path(output_path).with_suffix(".raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(raw_output_path),
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
                labels_to_draw = []

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

                    if (
                        plate_recognizer is not None
                        and track_id is not None
                    ):
                        self._maybe_read_plate(
                            plate_recognizer,
                            frame,
                            (x1, y1, x2, y2),
                            track_id,
                            frame_number,
                            track_plates,
                            plate_retry_interval,
                            plate_confidence_threshold,
                            max_plate_attempts
                        )

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Base label is always ASCII, so cv2's built-in font
                    # (fast, no extra rendering pass) handles it fine
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

                    # Plate text may contain Devanagari, which cv2's font
                    # can't render (it silently substitutes '?'), so it's
                    # queued for a separate PIL pass with a bundled font
                    # that can. Reserve the separator in the cv2-drawn
                    # base label so it isn't just a blank gap.
                    plate_entry = (
                        track_plates.get(track_id)
                        if track_id is not None else None
                    )
                    has_plate_text = bool(
                        plate_entry and plate_entry["text"]
                    )

                    if has_plate_text:
                        label += " |"

                    label_pos = (x1, max(y1 - 10, 20))

                    cv2.putText(
                        frame,
                        label,
                        label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    if has_plate_text:
                        (label_width, _), _ = cv2.getTextSize(
                            label + " ",
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            2
                        )
                        labels_to_draw.append((
                            label_pos[0] + label_width,
                            label_pos[1] - 18,
                            plate_entry["text"]
                        ))

                if labels_to_draw:
                    frame = self._draw_labels(frame, labels_to_draw)

            writer.write(frame)

        cap.release()
        writer.release()

        if self._transcode_to_h264(raw_output_path, output_path):
            raw_output_path.unlink(missing_ok=True)
        else:
            shutil.move(str(raw_output_path), output_path)

        plates = {
            str(track_id): {
                "text": entry["text"],
                "confidence": entry["confidence"]
            }
            for track_id, entry in track_plates.items()
            if entry["text"]
        }

        return {
            "frames_processed": frame_number,
            "unique_vehicles": len(unique_vehicle_ids),
            "output_path": output_path,
            "plates": plates
        }

    def _draw_labels(self, frame, labels_to_draw):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        for x, y, text in labels_to_draw:
            draw.text((x, y), text, font=self._font, fill=(0, 255, 0))

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _transcode_to_h264(self, source_path: Path, dest_path: str) -> bool:
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(source_path),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    dest_path
                ],
                check=True,
                capture_output=True
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            # ffmpeg missing or transcode failed — caller falls back to
            # the raw mp4v file (downloadable/playable outside browsers,
            # just not inline in a <video> element)
            return False

    def _maybe_read_plate(
        self,
        plate_recognizer,
        frame,
        vehicle_box,
        track_id,
        frame_number,
        track_plates,
        retry_interval,
        confidence_threshold,
        max_attempts
    ):
        entry = track_plates.get(track_id)

        if entry is None:
            should_attempt = True
        else:
            good_enough = entry["confidence"] >= confidence_threshold
            out_of_attempts = entry["attempts"] >= max_attempts
            due_for_retry = (
                frame_number - entry["last_attempt_frame"]
                >= retry_interval
            )
            should_attempt = (
                not good_enough
                and not out_of_attempts
                and due_for_retry
            )

        if not should_attempt:
            return

        x1, y1, x2, y2 = vehicle_box
        vehicle_crop = frame[max(y1, 0):y2, max(x1, 0):x2]

        best_text = entry["text"] if entry else None
        best_confidence = entry["confidence"] if entry else 0.0

        if vehicle_crop.size > 0:
            plates = plate_recognizer.read_plates(vehicle_crop)

            for plate in plates:
                if (
                    plate["text"]
                    and plate["ocr_confidence"] is not None
                    and plate["ocr_confidence"] > best_confidence
                ):
                    best_text = plate["text"]
                    best_confidence = plate["ocr_confidence"]

        track_plates[track_id] = {
            "text": best_text,
            "confidence": best_confidence,
            "attempts": (entry["attempts"] if entry else 0) + 1,
            "last_attempt_frame": frame_number
        }