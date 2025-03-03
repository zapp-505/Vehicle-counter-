from ultralytics import YOLO
import cv2
import torch

# Load YOLOv8 model with tracking enabled
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# Set your video path
video_path = "/content/PXL_20250217_031521987~2[1] (1).mp4"

# Open video file
cap = cv2.VideoCapture(video_path)

# Dictionary to store tracked vehicle IDs
tracked_vehicles = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Track objects using YOLO's built-in ByteTrack
    results = model.track(frame, persist=True, device=device)  

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get detected class ID
            track_id = int(box.id[0]) if box.id is not None else None  # Get tracking ID

            # Count only vehicles (Ignore humans and other objects)
            if class_id in [2, 3, 5, 7] and track_id is not None:
                tracked_vehicles.add(track_id)  # Store unique vehicle ID

cap.release()

print(f"Total unique vehicles detected: {len(tracked_vehicles)}")
