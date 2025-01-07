from utils import read_video, save_video
from trackers.player_tracker import PlayerTracker  # Import PlayerTracker directly

def main():
    # Read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect players
    player_tracker = PlayerTracker(model_path='yolov5n.pt')  # Renamed variable to 'player_tracker'
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )

    # Draw Output
    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    # Save video
    save_video(output_video_frames, output_video_path="output_videos/output_video.avi")

if __name__ == "__main__":
    main()
