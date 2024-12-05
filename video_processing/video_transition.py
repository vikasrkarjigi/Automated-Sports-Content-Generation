import cv2
import numpy as np

def create_transition(cap, start_frame, end_frame, transition_type='fade', duration_frames=30):
    """
    Create a transition between two frames in a video.

    Parameters:
    cap: cv2.VideoCapture object
    start_frame: int, starting frame number
    end_frame: int, ending frame number
    transition_type: str, type of transition ('fade', 'wipe_left', 'wipe_right', 'dissolve')
    duration_frames: int, number of frames for the transition

    Returns:
    list of frames containing the transition
    """
    # Save original position
    original_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Get the two frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
    ret, frame2 = cap.read()

    if not ret or frame1 is None or frame2 is None:
        raise ValueError("Could not read frames")

    # Convert frames to float32 for better transition quality
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)

    transition_frames = []

    for i in range(duration_frames):
        progress = i / (duration_frames - 1)

        if transition_type == 'fade':
            # Simple fade transition
            frame = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)

        elif transition_type == 'wipe_left':
            # Wipe from left to right
            width = frame1.shape[1]
            cut_point = int(width * progress)
            frame = frame1.copy()
            frame[:, :cut_point] = frame2[:, :cut_point]

        elif transition_type == 'wipe_right':
            # Wipe from right to left
            width = frame1.shape[1]
            cut_point = int(width * (1 - progress))
            frame = frame1.copy()
            frame[:, cut_point:] = frame2[:, cut_point:]

        elif transition_type == 'dissolve':
            # Dissolve with random pixels
            mask = np.random.random(frame1.shape[:2]) < progress
            mask = np.stack([mask] * 3, axis=2)
            frame = np.where(mask, frame2, frame1)

        else:
            raise ValueError(f"Unknown transition type: {transition_type}")

        # Convert back to uint8 for display
        frame_uint8 = frame.astype(np.uint8)
        transition_frames.append(frame_uint8)

    # Restore original position
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    return transition_frames


def main():
    # Open the video file
    cap = cv2.VideoCapture('video_processing/results/output_video.mp4')

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        # Define frames for significant events (modify these according to the video duration)
        game_start_frame = 100  # Game start frame
        new_set_frame = 300  # Frame where a new set starts
        point_scored_frame = 500  # Frame where a point is scored

        # Define the next event frame (new set or point scored)
        next_event_frame = 200

        # Create transitions for multiple events
        transitions = []
        transitions.append(
            create_transition(cap, game_start_frame, next_event_frame, transition_type='fade', duration_frames=30))
        transitions.append(
            create_transition(cap, new_set_frame, new_set_frame + 100, transition_type='wipe_left', duration_frames=30))
        transitions.append(
            create_transition(cap, point_scored_frame, point_scored_frame + 100, transition_type='dissolve',
                              duration_frames=30))

        # Create video writer for saving the transition
        first_frame = transitions[0][0]
        out = cv2.VideoWriter('highlight_with_transitions.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              30,
                              (first_frame.shape[1], first_frame.shape[0]))

        # Loop through the video and write frames to output
        current_transition_index = 0
        transition_frames = []
        current_frame_pos = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Apply the current transition only if we are at the event frame
            if current_frame_pos == game_start_frame and current_transition_index == 0:
                transition_frames = transitions[current_transition_index]
                current_transition_index += 1

            elif current_frame_pos == new_set_frame and current_transition_index == 1:
                transition_frames = transitions[current_transition_index]
                current_transition_index += 1

            elif current_frame_pos == point_scored_frame and current_transition_index == 2:
                transition_frames = transitions[current_transition_index]
                current_transition_index += 1

            # If a transition is being applied, write the transition frames
            if transition_frames:
                for transition_frame in transition_frames:
                    out.write(transition_frame)
                transition_frames = []  # Reset transition frames once written

            # Write the current frame to the output video
            out.write(frame)

            # If we've processed all frames, break the loop
            if current_frame_pos >= total_frames:
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        print("Transitions applied successfully to the video.")
        # Clean up resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
