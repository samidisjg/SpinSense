from utils import read_video, save_video
from trackers.player_tracker import PlayerTracker  # Import PlayerTracker directly
from trackers.ball_tracker import BallTracker  # Import BallTracker directly
from court_line_detector.court_line_detector import CourtLineDetector  # Import CourtLineDetector directly

def main():
    # Read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect players X and Ball
    player_tracker = PlayerTracker(model_path='yolov5n.pt')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )
    
    # court line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # print("Predicted court keypoints:", court_keypoints)



    # Draw Output
    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    # output_video_frames = []
    # for frame in video_frames:
    #     court_keypoints = court_line_detector.predict(frame)
    #     frame_with_keypoints = court_line_detector.draw_keypoints(frame, court_keypoints)
    #     output_video_frames.append(frame_with_keypoints)


    # Save video
    save_video(output_video_frames, output_video_path="output_videos/output_video.avi")

if __name__ == "__main__":
    main()
