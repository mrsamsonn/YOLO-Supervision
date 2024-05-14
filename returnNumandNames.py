from ultralytics import YOLO
import cv2
import time


model = YOLO("yolov8n.pt")  # Load the YOLO model
names = model.names

def print_detections(results,previous_detections):

    new_detections = set()  # Store the objects detected in the current frame
    
    # Initialize variables
    prev_name = None
    count = 0
    new_detections = set()

    for detection in results.boxes.cls:
        # Assuming single class detection and using the most probable class
        name = names[int(detection)]

        # Check if the current name is the same as the previous one
        if prev_name == name:
            count += 1
        else:
            # If the name changes, print the previous name and count
            if prev_name:
                print(prev_name, ", ", count)
            # Reset count for the new name
            count = 1
        
        # Add the current name to the set of new detections
        new_detections.add(name)
        
        # Update the previous name
        prev_name = name

    # Print the last name and count
    if prev_name:
        print(prev_name, ", ", count)


    return new_detections


# Use stream=True for video streams (replace with source if using an image)
results = model(source="0", conf=0.6 , stream=True, show=True, verbose=False)  # Assuming webcam as source

previous_detections = set()
start_time = time.time()

for frame in results:
     if time.time() - start_time >= 1:  # Check for new detections every [] seconds
        previous_detections = print_detections(frame, previous_detections)  # Pass each frame's detection results
        previous_detections = previous_detections  # Reset previous_detections for next frame
        start_time = time.time()  # Reset the start time for the next 2-second interval


# Release resources (if using video stream)
cv2.destroyAllWindows()
