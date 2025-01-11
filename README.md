# SpinSense 🏓  
**AI/ML-Powered Tennis Analysis System**  

SpinSense is an AI/ML-driven tennis analysis system that combines state-of-the-art computer vision and deep learning techniques to analyze tennis gameplay. Using YOLO for object detection, PyTorch for keypoint extraction, and object tracking, this system provides a comprehensive toolset for understanding player and ball dynamics. ⚾ 

---

## Features 🚀🎾  
### Object Detection with YOLOv5n (Ultralytics) 
- Detect tennis players and balls in videos using the YOLOv5 object detection model.  
- Fine-tune YOLOv5n on custom dataset for tennis ball and player detection 

### Object Tracking  
- Implement object trackers to maintain consistent tracking of players and tennis balls across video frames.  

### Keypoint Detection with PyTorch  
- Train a convolutional neural network (CNN) to extract key points for identifying court boundaries and features.

### OpenCV (CV2)
- Use for video manipulation and processing.

### Tennis Analyzer  
- Combine all detection and tracking data to deliver actionable insights about the gameplay.  
- Analyze and visualize data for performance improvements.  

---

## Technologies Used 🛠️🎾
- YOLOv5n for object detection.
- PyTorch for training custom CNNs for keypoint detection.
- OpenCV for video processing and visualization.
- NumPy for data analysis and manipulation.   

---

## Future Improvements 🗺️🔮
- Add support for real-time video feeds.
- Extend the analysis to other sports like table tennis or basketball.
- Incorporate advanced analytics like spin, speed, and shot prediction.

## Installation 📌🕹️ 
1. Clone the repository:  
   ```bash  
   git clone https://github.com/samidisjg/SpinSense.git  
   cd SpinSense
 
2. Download YOLOv5n weights from the Ultralytics YOLOv5n GitHub repository.

   <img src="/input_videos/output.png" alt="tennis_court" style="width: 100%;">

## Copyright
© 2025 SpinSense. All rights reserved.








