import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from collections import defaultdict

# Constants
EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_BLINK_FRAMES = 3

# Initialize variables
blink_counter = 0
total_blinks = 0
gaze_outside_frame_count = 0

# Event timestamps
event_logs = defaultdict(list)

# Initialize Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/shravansrinivasan/Downloads/gen ai project/shape_predictor_68_face_landmarks.dat")

# Extract eye landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) to detect blinking."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_eye_region(shape, eye_points):
    """Extract the eye region from the facial landmarks."""
    return np.array([(shape.part(point).x, shape.part(point).y) for point in eye_points], dtype=np.int32)

def detect_yawn(shape):
    """Detect yawns based on mouth opening."""
    vertical_distance = dist.euclidean(shape.part(62), shape.part(66))
    horizontal_distance = dist.euclidean(shape.part(48), shape.part(54))
    return vertical_distance / horizontal_distance > 0.5  # Yawn threshold

def process_video(video_path):
    global blink_counter, total_blinks, gaze_outside_frame_count

    # Initialize yawn counter
    yawn_count = 0

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_counter = 0  # To track frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        frame_counter += 1
        timestamp = frame_counter / fps  # Calculate timestamp in seconds

        for face in faces:
            shape = predictor(gray, face)

            # Extract left and right eye regions
            left_eye = get_eye_region(shape, LEFT_EYE)
            right_eye = get_eye_region(shape, RIGHT_EYE)

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0

            # Blink detection
            if avg_ear < EYE_ASPECT_RATIO_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EYE_BLINK_FRAMES:
                    total_blinks += 1
                    event_logs['blinks'].append(timestamp)
                blink_counter = 0

            # Gaze estimation
            left_gaze_x, _, _ = calculate_gaze(left_eye, frame)
            right_gaze_x, _, _ = calculate_gaze(right_eye, frame)

            if left_gaze_x and right_gaze_x:
                if (left_gaze_x < np.min(left_eye[:, 0]) or
                        left_gaze_x > np.max(left_eye[:, 0]) or
                        right_gaze_x < np.min(right_eye[:, 0]) or
                        right_gaze_x > np.max(right_eye[:, 0])):
                    gaze_outside_frame_count += 1
                    event_logs['gaze_outside'].append(timestamp)

            # Yawn detection
            if detect_yawn(shape):
                yawn_count += 1
                event_logs['yawns'].append(timestamp)

        # Show the frame
        cv2.imshow("Eye Blink and Gaze Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return all relevant metrics and event logs
    return total_blinks, gaze_outside_frame_count, yawn_count, event_logs


def analyze_events(event_logs):
    """Analyze events to find periods with the most activity."""
    analysis_results = {}
    for event_type, timestamps in event_logs.items():
        # Group timestamps into 1-second bins
        bins = defaultdict(int)
        for timestamp in timestamps:
            time_bin = int(timestamp)  # Convert to integer second
            bins[time_bin] += 1

        # Find the time bin with the most events
        max_time_bin = max(bins, key=bins.get, default=None)
        max_count = bins[max_time_bin] if max_time_bin is not None else 0

        analysis_results[event_type] = {
            "most_active_second": max_time_bin,
            "event_count": max_count,
        }

    return analysis_results


if __name__ == "__main__":
    video_path = "/Users/shravansrinivasan/Downloads/gen ai project/5439068-uhd_2560_1440_25fps.mp4"  # Replace with your video path
    blinks, gaze_outs, yawns, logs = process_video(video_path)

    print(f"Total Blinks: {blinks}")
    print(f"Total Gaze Outside Frame: {gaze_outs}")
    print(f"Total Yawns: {yawns}")

    # Analyze events
    results = analyze_events(logs)
    for event, data in results.items():
        print(f"{event.capitalize()}: Most active at second {data['most_active_second']} with {data['event_count']} events.")
