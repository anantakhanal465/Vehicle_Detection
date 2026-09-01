from app.services.vehicle_detector import VehicleDetector


detector = VehicleDetector()

image = "bus.jpg"

detections = detector.detect(
    image,
    vehicle_type="car"
)

print("\nDetected Vehicles:")
print("------------------")

for detection in detections:
    print(detection)