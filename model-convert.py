from ultralytics import YOLO

model = YOLO("yolov8n.pt")



tfLite = model.export(format="openvino")




# pred = YOLO("./yolov8n_openvino_model")

# results = pred.predict(source="2", show=True)

# print(results)