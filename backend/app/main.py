from fastapi import FastAPI

from app.api.detection import router as detection_router


app = FastAPI(
    title="Vehicle Detection API",
    description="YOLO-based vehicle and license plate detection system",
    version="1.0.0"
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