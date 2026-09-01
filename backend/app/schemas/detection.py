from typing import Literal

from pydantic import BaseModel


class DetectionResponse(BaseModel):
    success: bool
    vehicle_type: str
    detections: list