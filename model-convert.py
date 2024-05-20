from ultralytics import YOLO

model = YOLO("pantryv2.pt")



tfLite = model.export(format="openvino")




# pred = YOLO("./yolov8n_openvino_model")

# results = pred.predict(source="2", show=True)

# print(results)