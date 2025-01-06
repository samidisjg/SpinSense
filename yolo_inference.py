from ultralytics import YOLO

model = YOLO('models/yolo5_last.pt')

# Image
result = model.predict('D:/projects done/SpinSense-/input_videos/input_video.mp4',conf=0.2,save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)