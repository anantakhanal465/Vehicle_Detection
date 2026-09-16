from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, JSON, String

from app.database import Base


class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )
    source_type = Column(String, nullable=False)
    vehicle_type = Column(String, nullable=False)
    detections = Column(JSON, nullable=True)
    unique_vehicles = Column(Integer, nullable=True)
    frames_processed = Column(Integer, nullable=True)
    download_url = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
