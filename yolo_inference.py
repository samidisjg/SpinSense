from ultralytics import YOLO

model = YOLO('yolov5n.pt')  # Load model

result = model.predict('D:/projects done/SpinSense-/input_videos/test_2.mp4',save=True)
print(result)

print("boxes:")
for box in result[0].boxes:
    print(box)
