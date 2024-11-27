import cv2
import numpy as np

def create_transition(cap, start_frame, end_frame, transition_type='fade', duration_frames=30):
    """
    Create a high-quality transition between two frames in a video.

    Parameters:
    cap: cv2.VideoCapture object
    start_frame: int, starting frame number
    end_frame: int, ending frame number
    transition_type: str, type of transition ('fade', 'wipe_left', 'wipe_right', 'dissolve', 'motion_blend')
    duration_frames: int, number of frames for the transition

    Returns:
    list of frames containing the transition
    """
    # Save the current position
    original_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Get the two frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret1, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
    ret2, frame2 = cap.read()

    if not ret1 or not ret2 or frame1 is None or frame2 is None:
        raise ValueError("Could not read the specified frames from the video.")

    # Convert frames to float32 for better transition quality
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)

    height, width, _ = frame1.shape
    transition_frames = []

    for i in range(duration_frames):
        progress = i / (duration_frames - 1)

        if transition_type == 'fade':
            # Improved fade with gamma correction
            frame = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
            frame = cv2.pow(frame / 255.0, 1.2) * 255.0  # Gamma correction

        elif transition_type == 'wipe_left':
            # Smooth gradient wipe from left to right
            cut_point = int(width * progress)
            gradient = np.linspace(0, 1, cut_point).reshape(1, -1, 1)
            gradient = np.tile(gradient, (height, 1, 3))
            frame = frame1.copy()
            frame[:, :cut_point] = frame1[:, :cut_point] * (1 - gradient) + frame2[:, :cut_point] * gradient

        elif transition_type == 'wipe_right':
            # Smooth gradient wipe from right to left
            cut_point = int(width * (1 - progress))
            gradient = np.linspace(0, 1, width - cut_point).reshape(1, -1, 1)
            gradient = np.tile(gradient, (height, 1, 3))
            frame = frame1.copy()
            frame[:, cut_point:] = frame1[:, cut_point:] * (1 - gradient) + frame2[:, cut_point:] * gradient

        elif transition_type == 'dissolve':
            # Non-linear dissolve using Perlin noise
            mask = (np.random.normal(loc=progress, scale=0.1, size=(height, width)) > 0.5).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)  # Smooth dissolve effect
            mask = np.stack([mask] * 3, axis=2)
            frame = frame1 * (1 - mask) + frame2 * mask

        elif transition_type == 'motion_blend':
            # Motion-aware blend using optical flow
            gray1 = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[:, :, 0] * progress, flow[:, :, 1] * progress
            map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
            map_x = (map_x + dx).astype(np.float32)
            map_y = (map_y + dy).astype(np.float32)
            warped_frame2 = cv2.remap(frame2, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            frame = cv2.addWeighted(frame1, 1 - progress, warped_frame2, progress, 0)

        else:
            raise ValueError(f"Unknown transition type: {transition_type}")

        # Convert back to uint8
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        transition_frames.append(frame_uint8)

    # Restore the original position
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    return transition_frames

def main():
    # Open the video file
    video_path = 'output_video.mp4'  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a video writer for saving the final video
    output_path = 'enhanced_transition_full_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Loop through the video, applying transitions between frames
    try:
        transition_type = 'fade'  # Try 'fade', 'wipe_left', 'wipe_right', 'dissolve', or 'motion_blend'
        transition_duration = 30  # Length of the transition in frames

        # Read frames and apply transitions
        prev_frame = None
        for i in range(total_frames - transition_duration):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                break

            # Apply transition between consecutive frames
            if prev_frame is not None:
                transition_frames = create_transition(
                    cap, i - transition_duration, i,
                    transition_type=transition_type,
                    duration_frames=transition_duration
                )

                # Write transition frames to output
                for transition_frame in transition_frames:
                    out.write(transition_frame)

            # Write the current frame (without transition)
            out.write(frame)
            prev_frame = frame

        print(f"Final video with transitions saved as '{output_path}'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Clean up resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
