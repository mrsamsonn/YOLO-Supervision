from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")  # Load the YOLO model
names = model.names

def print_detections(results, previous_counts):

    new_detections = set()  # Store the objects detected in the current frame
    
    # Initialize variables
    current_counts = {}

    for detection in results.boxes.cls:
        # Assuming single class detection and using the most probable class
        name = names[int(detection)]

        # Increment count for the current class
        current_counts[name] = current_counts.get(name, 0) + 1

    # Check for changes in counts and print
    for name, count in current_counts.items():
        if name not in previous_counts or count != previous_counts[name]:
            print(name, ", ", count)
    
    # Check for classes that were detected previously but not anymore
    for name in previous_counts:
        if name not in current_counts:
            print(name, ", ", 0)  # Update count to 0 for classes no longer detected

    return new_detections, current_counts


# Use stream=True for video streams (replace with source if using an image)
results = model(source="0", conf=0.6 , stream=True, show=True, verbose=False)  # Assuming webcam as source

previous_counts = {}

start_time = time.time()

for frame in results:
    if time.time() - start_time >= 1:  # Check for new detections every 1 second
        _, previous_counts = print_detections(frame, previous_counts)  # Pass each frame's detection results
        start_time = time.time()  # Reset the start time for the next 1-second interval

# Release resources (if using video stream)
cv2.destroyAllWindows()
