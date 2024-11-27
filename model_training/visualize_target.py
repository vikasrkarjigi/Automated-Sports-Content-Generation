import cv2
import pandas as pd
import argparse
from pathlib import Path


def load_frame_values(csv_path):
    """
    Load frame values from CSV file into a dictionary.
    If a frame doesn't exist in the CSV, it will return 0 by default.
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df['frame'], df['value']))


def process_video(video_path, csv_path, output_path=None):
    """
    Process video file and overlay values from CSV on each frame.
    """
    # Load frame values
    frame_values = load_frame_values(csv_path)

    # Get the first frame number from the CSV
    first_frame = min(frame_values.keys())

    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    frame_number = first_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get value for current frame (default to 0 if not in CSV)
        value = frame_values.get(frame_number, 0)

        # Add text overlay
        text = f"Frame: {frame_number}, Value: {value}"
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display frame
        cv2.imshow('Video with Values', frame)

        # Write frame if output path is provided
        if output_path:
            out.write(frame)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Overlay CSV values on video frames')
    parser.add_argument('video_path', type=Path, help='Path to input video file')
    parser.add_argument('csv_path', type=Path, help='Path to CSV file with frame values')
    parser.add_argument('--output', '-o', type=Path, help='Optional path for output video')

    args = parser.parse_args()

    # Verify input files exist
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    process_video(args.video_path, args.csv_path, args.output)


if __name__ == "__main__":
    main()