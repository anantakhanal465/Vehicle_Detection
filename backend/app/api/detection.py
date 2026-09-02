import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

import cv2
import numpy as np

from app.database import get_db_session
from app.models import DetectionRecord
from app.schemas.detection import DetectionResponse
from app.services.plate_recognizer import PlateRecognizer
from app.services.vehicle_detector import VehicleDetector
from app.services.video_processor import VideoProcessor


router = APIRouter(
    prefix="/api/detection",
    tags=["Detection"]
)

detector = VehicleDetector()
video_processor = VideoProcessor()
plate_recognizer = PlateRecognizer()

# Fraction of a plate box's area that must fall inside a vehicle box
# for the plate to be attributed to that vehicle
PLATE_MATCH_THRESHOLD = 0.5

BACKEND_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _match_plate_to_vehicle(plate_box, vehicles):
    plate_area = (
        max(0, plate_box["x2"] - plate_box["x1"])
        * max(0, plate_box["y2"] - plate_box["y1"])
    )

    if plate_area == 0:
        return None

    best_vehicle = None
    best_overlap = 0.0

    for vehicle in vehicles:
        vb = vehicle["bounding_box"]

        ix1 = max(plate_box["x1"], vb["x1"])
        iy1 = max(plate_box["y1"], vb["y1"])
        ix2 = min(plate_box["x2"], vb["x2"])
        iy2 = min(plate_box["y2"], vb["y2"])

        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        overlap_ratio = intersection / plate_area

        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_vehicle = vehicle

    return best_vehicle if best_overlap >= PLATE_MATCH_THRESHOLD else None


@router.post("/image", response_model=DetectionResponse)
async def detect_image(
    vehicle_type: Literal[
        "all",
        "car",
        "motorcycle",
        "bus",
        "truck"
    ] = Form("all"),

    file: UploadFile = File(...)
):
    # Read uploaded image
    contents = await file.read()

    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Run YOLO
    detections = detector.detect(
        image,
        vehicle_type=vehicle_type
    )

    with get_db_session() as db:
        db.add(DetectionRecord(
            source_type="image",
            vehicle_type=vehicle_type,
            detections=detections
        ))
        db.commit()

    return {
        "success": True,
        "vehicle_type": vehicle_type,
        "detections": detections
    }


@router.post("/plate")
async def detect_plate(file: UploadFile = File(...)):
    contents = await file.read()

    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    vehicles = detector.detect(image, vehicle_type="all")
    plates = plate_recognizer.read_plates(image)

    for vehicle in vehicles:
        vehicle["license_plate"] = None

    unmatched_plates = []

    for plate in plates:
        vehicle = _match_plate_to_vehicle(plate["bounding_box"], vehicles)

        if vehicle is not None:
            vehicle["license_plate"] = plate
        else:
            unmatched_plates.append(plate)

    with get_db_session() as db:
        db.add(DetectionRecord(
            source_type="plate",
            vehicle_type="all",
            detections=vehicles
        ))
        db.commit()

    return {
        "success": True,
        "vehicles": vehicles,
        "unmatched_plates": unmatched_plates
    }


@router.post("/video")
async def detect_video(
    vehicle_type: Literal[
        "all",
        "car",
        "motorcycle",
        "bus",
        "truck"
    ] = Form("all"),

    read_plates: bool = Form(False),

    file: UploadFile = File(...)
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
            )
        )

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{extension}"
    output_path = RESULTS_DIR / f"{job_id}_tracked.mp4"

    # Save uploaded video to disk (cv2.VideoCapture needs a file path)
    contents = await file.read()
    input_path.write_bytes(contents)

    try:
        result = await run_in_threadpool(
            video_processor.process,
            input_path=str(input_path),
            output_path=str(output_path),
            vehicle_type=vehicle_type,
            plate_recognizer=plate_recognizer if read_plates else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        input_path.unlink(missing_ok=True)

    download_url = f"/api/detection/video/{output_path.name}"
    plates = result.get("plates", {})

    with get_db_session() as db:
        db.add(DetectionRecord(
            source_type="video",
            vehicle_type=vehicle_type,
            detections=plates or None,
            frames_processed=result["frames_processed"],
            unique_vehicles=result["unique_vehicles"],
            download_url=download_url
        ))
        db.commit()

    return {
        "success": True,
        "vehicle_type": vehicle_type,
        "frames_processed": result["frames_processed"],
        "unique_vehicles": result["unique_vehicles"],
        "download_url": download_url,
        "plates": plates
    }


@router.get("/history")
async def get_history(limit: int = 20):
    with get_db_session() as db:
        records = (
            db.query(DetectionRecord)
            .order_by(DetectionRecord.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": record.id,
                "created_at": record.created_at.isoformat(),
                "source_type": record.source_type,
                "vehicle_type": record.vehicle_type,
                "detections": record.detections,
                "frames_processed": record.frames_processed,
                "unique_vehicles": record.unique_vehicles,
                "download_url": record.download_url
            }
            for record in records
        ]


@router.get("/video/{filename}")
async def download_video(filename: str):
    # Reject anything that isn't a bare filename to prevent path traversal
    if filename != Path(filename).name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = RESULTS_DIR / filename

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )