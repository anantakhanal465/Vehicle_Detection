from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.detection import router as detection_router
from app.database import Base, engine, ensure_schema
from app import models  # noqa: F401 (registers models with Base's metadata)


app = FastAPI(
    title="Vehicle Detection API",
    description="YOLO-based vehicle and license plate detection system",
    version="1.0.0"
)

Base.metadata.create_all(bind=engine)
ensure_schema()

# Permissive for local development; restrict to specific origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


app.include_router(detection_router)


@app.get("/")
def root():
    return {
        "message": "Vehicle Detection API is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    } 