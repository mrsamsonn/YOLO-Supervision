from roboflow import Roboflow
rf = Roboflow(api_key="18PZpI8vgWrbkC6iYYvG")
project = rf.workspace("samrat-sahoo").project("groceries-6pfog")
version = project.version(7)
dataset = version.download("yolov8")
