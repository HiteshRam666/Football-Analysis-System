from ultralytics import YOLO

model = YOLO("D:\\Code\\Football\\models\\best.pt")
results = model.predict("D:\\Code\\Football\\input_videos\\08fd33_4.mp4")
print(results[0])
print("=========================")
for box in results[0].boxes:
    print(box)
