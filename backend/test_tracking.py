from ultralytics import YOLO


model = YOLO("yolo11n.pt")


results = model.track(
    source="videos/traffic1.mp4",
    tracker="bytetrack.yaml",
    classes=[2, 3, 5, 7],
    save=True,
    show=False
)


print("Tracking completed.")