from ultralytics import YOLO
import cv2
import time


model = YOLO("best.pt")  # Load the YOLO model

def print_detections(results,food,previous_detections, left_boundary):

    new_detections = set()  # Store the objects detected in the current frame
  
    for detection in results:
        # Assuming single class detection and using the most probable class
        name = detection.names[0]
        new_detections.add(name)
        center_x = (detection.boxes.data[0][0] + detection.boxes.data[0][2]) / 2  # Calculate center x-coordinate
        print("coordinates: ",center_x)

        if name not in previous_detections:
            if center_x < left_boundary:  # Check if object is on the left side
                food += 1
                print(f"Detected object: {name} on the left side")
                print("intFood: ", food)
            else:
                food -= 1
                print(f"Detected object: {name} on the right side")
                print("intFood: ", food)

    return food, new_detections


# Use stream=True for video streams (replace with source if using an image)
results = model(source="0", conf=0.6 , stream=True, show=True, verbose=False)  # Assuming webcam as source

food = 0
previous_detections = set()
start_time = time.time()
left_boundary = 320  # Assuming the left boundary is at the center of the frame

for frame in results:
     if time.time() - start_time >= 1:  # Check for new detections every [] seconds
        food, previous_detections = print_detections(frame, food, previous_detections, left_boundary)  # Pass each frame's detection results
        previous_detections = previous_detections  # Reset previous_detections for next frame
        start_time = time.time()  # Reset the start time for the next 2-second interval


# Release resources (if using video stream)
cv2.destroyAllWindows()
