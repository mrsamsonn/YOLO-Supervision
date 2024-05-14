from roboflow import Roboflow
rf = Roboflow(api_key="18PZpI8vgWrbkC6iYYvG")
project = rf.workspace("yolo-1eope").project("ortega-diced-green-chiles")
version = project.version(2)
dataset = version.download("yolov8")
