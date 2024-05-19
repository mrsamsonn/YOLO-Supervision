from roboflow import Roboflow
rf = Roboflow(api_key="18PZpI8vgWrbkC6iYYvG")
project = rf.workspace("yolo-1eope").project("pantry-wlooe")
version = project.version(1)
dataset = version.download("yolov8")
