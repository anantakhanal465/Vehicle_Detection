# Vehicle Detection & License Plate Recognition (Nepal)

FastAPI backend for detecting vehicles in images/video, tracking them
across frames, and reading license plates — tuned for Nepal, with
EasyOCR configured for both Devanagari-script and embossed Latin
plates.

## Stack

- **YOLO11** (`ultralytics`) for vehicle detection (car/motorcycle/bus/truck)
- **ByteTrack** for multi-object tracking across video frames
- **YOLOv8 license-plate detector**, as an ensemble of two models (see below) — the base one is [Koushim/yolov8-license-plate-detection](https://huggingface.co/Koushim/yolov8-license-plate-detection) (MIT)
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

`yolo11n.pt` (vehicle detector) and the plate-detector weights are not
committed to the repo — the vehicle model auto-downloads via
`ultralytics` on first run. `PlateRecognizer` loads **two** plate
models and merges their detections (see "Plate detector ensemble"
below); both need to be fetched manually:

```bash
curl -sL -o backend/license_plate_detector.pt \
  "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"
```

The second model, `license_plate_detector_nepal.pt`, is a fine-tune of
the first on Nepali plates — it isn't published anywhere, so regenerate
it with `backend/scripts/finetune_plate_detector.py` (see that file's
docstring for the dataset setup; takes ~35-45 min on CPU) and copy its
output weights to `backend/license_plate_detector_nepal.pt`. Without
this second file, `PlateRecognizer`'s `YOLO(...)` load will fail — either
run the fine-tune or edit `DEFAULT_MODEL_PATHS` in
`app/services/plate_recognizer.py` down to just the first model.

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

## Plate detector ensemble

`PlateRecognizer` runs **both** plate models on every image and merges
their detections with IoU-based NMS (overlapping boxes from each model
collapse into one, keeping the higher-confidence box; non-overlapping
boxes from either model are both kept). This wasn't the original plan —
here's why it ended up this way, since the reasoning matters if you're
tuning this further:

1. A full fine-tune of the base model on ~8,000 Nepali plate photos
   (20 epochs, default LR, nothing frozen) reached mAP50 0.98 on its own
   held-out validation split — but that split came from the same narrow
   domain as training (close-range phone photos, plate filling 20-40%
   of frame). Tested against real aerial traffic footage instead, it
   fixated on a background storefront sign as a false positive on
   *every single frame* — it had learned "red rectangle = plate" from a
   domain where that heuristic always held, and had no counterexamples.
2. A much gentler fine-tune (3 epochs, low LR, frozen backbone —
   `scripts/finetune_plate_detector.py`) avoided that regression. But
   tested against the same footage, it wasn't a strict improvement over
   the original either — it reliably found different real plates than
   the original does, missing at least one the original always caught.
3. Rather than pick a winner, both run and their results get merged.
   Verified against real footage: this catches more real plates than
   either model alone, with no false positives observed in testing.

This isn't rigorously validated against a large labeled test set —
just checked by hand against several real frames from
`backend/videos/nepal_test.mp4`. Treat it as a reasonable default, not
a proven-optimal one.

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
