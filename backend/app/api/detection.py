from typing import Literal

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

import cv2
import numpy as np

from app.services.vehicle_detector import VehicleDetector


router = APIRouter(
    prefix="/api/detection",
    tags=["Detection"]
)

detector = VehicleDetector()


@router.post("/image")
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

    return {
        "success": True,
        "vehicle_type": vehicle_type,
        "detections": detections
    }