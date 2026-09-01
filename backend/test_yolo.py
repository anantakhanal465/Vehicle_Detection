from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Run detection on an image
results = model("https://ultralytics.com/images/bus.jpg")

# Display results
for result in results:
    result.show()