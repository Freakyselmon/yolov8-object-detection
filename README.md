# yolov8-object-detection
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Provide the correct file path for your video
video_path = 0  # Use 0 for webcam

# Open the video capture device with the appropriate backend
cap = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)  # Use CAP_DSHOW on Windows

# Check if the video source was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video source '{video_path}'.")
    print("Ensure the webcam is connected or the video file path is correct.")
    exit()

# Get video properties (for debugging)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video source opened successfully. Width: {width}, Height: {height}, FPS: {fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame or end of video.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)  # Perform inference

    # Iterate over the results to extract bounding boxes and labels
    for result in results:
        boxes = result.boxes  # Get the bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get bounding box coordinates
            conf = box.conf.item()  # Get confidence score
            cls = box.cls.item()  # Get class ID

            # Convert coordinates to integer
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Get the label and confidence value
            label = f'{model.names[int(cls)]} {conf:.2f}'  # Class name and confidence

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the annotated frame
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

print(f"Frame shape: {frame.shape}")  # Check if the frame is being captured
print(f"Results: {results}")  # Check the output of YOLOv8 inference
