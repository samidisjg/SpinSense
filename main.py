from utils import read_video, save_video
from trackers.player_tracker import PlayerTracker  # Import PlayerTracker directly
from trackers.ball_tracker import BallTracker  # Import BallTracker directly

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


    # Draw Output
    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)

    # Save video
    save_video(output_video_frames, output_video_path="output_videos/output_video.avi")

if __name__ == "__main__":
    main()
