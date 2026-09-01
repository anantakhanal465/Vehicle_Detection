from app.services.video_processor import VideoProcessor


processor = VideoProcessor()

result = processor.process(
    input_path="videos/nepal_test.mp4",
    output_path="results/nepal_tracked.mp4",
    vehicle_type="car"
)

print("\nNepal video processing completed")
print("--------------------------------")
print(f"Frames processed: {result['frames_processed']}")
print(f"Unique vehicles: {result['unique_vehicles']}")
print(f"Output: {result['output_path']}")