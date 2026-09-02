# Vehicle Detection & License Plate Recognition (Nepal)

FastAPI backend for detecting vehicles in images/video, tracking them
across frames, and reading license plates — tuned for Nepal, with
EasyOCR configured for both Devanagari-script and embossed Latin
plates.

## Stack

- **YOLO11** (`ultralytics`) for vehicle detection (car/motorcycle/bus/truck)
- **ByteTrack** for multi-object tracking across video frames
- **YOLOv8 license-plate detector** ([Koushim/yolov8-license-plate-detection](https://huggingface.co/Koushim/yolov8-license-plate-detection), MIT) for plate localization
- **EasyOCR** (`ne` + `en`) for reading plate text
- **FastAPI** + **SQLAlchemy** (SQLite by default, swappable to Postgres)
- **ffmpeg** (system binary) to transcode processed video output to H.264 — see note below
- **React + Vite + TypeScript + Tailwind** frontend

## Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU-only torch build first (avoids pulling unnecessary CUDA deps
# on machines without an NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cd backend
uvicorn app.main:app --reload
```

`yolo11n.pt` (vehicle detector) and `license_plate_detector.pt` (plate
detector) are not committed to the repo — the vehicle model
auto-downloads via `ultralytics` on first run; the plate model needs
to be fetched manually:

```bash
curl -sL -o backend/license_plate_detector.pt \
  "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"
```

By default, detection history is stored in a local SQLite file
(`backend/detection_history.db`). To use Postgres instead, set
`DATABASE_URL` (e.g. in a `.env` file in `backend/`):

```
DATABASE_URL=postgresql://user:password@localhost/vehicle_detection
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The dev server proxies `/api` to `http://127.0.0.1:8000` (see
`vite.config.ts`), so it expects the backend already running on that
port. Open the printed local URL (typically `http://127.0.0.1:5173`).

## API

All endpoints are under `/api/detection`. Interactive docs at `/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/image` | POST | Detect vehicles in an uploaded image. `vehicle_type` filter: `all`/`car`/`motorcycle`/`bus`/`truck`. |
| `/plate` | POST | Detect vehicles + license plates in an image, matched by bounding-box overlap and OCR'd. |
| `/video` | POST | Detect + track vehicles in an uploaded video, annotate and save the result. Set `read_plates=true` to also OCR each tracked vehicle's plate (slower — see below). |
| `/video/{filename}` | GET | Download a processed video by filename. |
| `/history` | GET | Recent detection records (`limit` query param, default 20). |
| `/health` | GET | Health check. |

## Notes & known limitations

- **OCR accuracy depends heavily on source resolution.** Plates in
  wide traffic-camera-style footage are often only tens of pixels
  wide — below what any OCR engine can reliably read, even with the
  upscaling + contrast enhancement this project applies. Real ANPR
  deployments use dedicated close-range cameras for this reason.
- **Video plate reading is throttled, not run every frame.** EasyOCR
  takes roughly 1-2 seconds per plate crop on CPU, so `read_plates=true`
  on video retries each tracked vehicle at most every 20 frames, stops
  once a confident reading is found, and caps at 5 attempts per
  vehicle (see `VideoProcessor.process` in
  `backend/app/services/video_processor.py`). Expect video processing
  to take noticeably longer with plate reading enabled.
- **Real-time streaming (websockets) is not implemented.** `websockets`
  is listed in `requirements.txt` for future use but nothing in the
  app uses it yet — everything is request/response today.
- Ultralytics `YOLO` model instances are not shared across services
  (`VehicleDetector`, `VideoProcessor`, `PlateRecognizer` each load
  their own) because they aren't safe for concurrent inference from
  multiple threads, which this API's threadpooled video processing
  can trigger.
- **Video output requires `ffmpeg` on `PATH`.** The `opencv-python`
  wheel installed by `requirements.txt` has no working H.264 encoder
  (no `libx264`, and its hardware-encoder fallback needs a device that
  usually doesn't exist), so `VideoProcessor` writes frames with
  OpenCV's MPEG-4 codec and then shells out to `ffmpeg` to transcode
  to H.264 — required for the output to play in any browser `<video>`
  element (MPEG-4 Part 2 is not a supported browser codec). If
  `ffmpeg` isn't installed, processing still succeeds but the returned
  file falls back to the non-H.264 original, which downloads fine but
  won't play inline in the frontend.
