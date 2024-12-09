import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from collections import defaultdict
import os
import requests
import openai
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found. Please set it in the .env file.")


BASE_URL = "https://api.perplexity.ai/chat/completions"  # Replace with the actual API endpoint

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
p = "data/raw/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)  # Update with the correct path

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
    vertical_distance = dist.euclidean((shape.part(62).x, shape.part(62).y), (shape.part(66).x, shape.part(66).y))
    horizontal_distance = dist.euclidean((shape.part(48).x, shape.part(48).y), (shape.part(54).x, shape.part(54).y))
    return vertical_distance / horizontal_distance > 0.5  # Yawn threshold

def calculate_gaze(eye, frame):
    """Estimate gaze direction based on eye region."""
    x_min, y_min = np.min(eye, axis=0)
    x_max, y_max = np.max(eye, axis=0)
    eye_frame = frame[y_min:y_max, x_min:x_max]

    if eye_frame.size == 0:  # Check if eye_frame is empty
        return None, None, None

    gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    _, thresh_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    # Find contours for the eye to locate the pupil
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) + x_min
            cy = int(M["m01"] / M["m00"]) + y_min
            return cx, cy, largest_contour
    return None, None, None



def process_video(video_path, max_time=60):
    """
    Process the video, but stop after max_time seconds.
    """
    global blink_counter, total_blinks, gaze_outside_frame_count

    # Initialize yawn counter
    yawn_count = 0

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps == 0:
        print("Error: Unable to open video or retrieve FPS.")
        return None

    frame_counter = 0  # To track frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        frame_counter += 1
        timestamp = frame_counter / fps  # Calculate timestamp in seconds

        # Stop processing if the timestamp exceeds max_time
        if timestamp > max_time:
            print(f"Reached maximum processing time: {max_time} seconds")
            break

        # Debugging: Check if faces are detected
        print(f"Frame {frame_counter}: Detected {len(faces)} face(s).")

        for face in faces:
            shape = predictor(gray, face)

            # Extract left and right eye regions
            left_eye = get_eye_region(shape, LEFT_EYE)
            right_eye = get_eye_region(shape, RIGHT_EYE)

            # Validate eye regions
            if left_eye.size == 0 or right_eye.size == 0:
                continue

            # Process gaze
            left_gaze_x, _, _ = calculate_gaze(left_eye, frame)
            right_gaze_x, _, _ = calculate_gaze(right_eye, frame)

            if left_gaze_x is None or right_gaze_x is None:
                continue

            # Blink detection
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EYE_ASPECT_RATIO_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EYE_BLINK_FRAMES:
                    total_blinks += 1
                    blink_counter = 0

            # Yawn detection
            if detect_yawn(shape):
                yawn_count += 1

        # Debugging: Log current counts
        print(f"Frame {frame_counter}: Total Blinks: {total_blinks}, Yawns: {yawn_count}")

        # Show the annotated frame
        cv2.imshow("Blink, Gaze, and Yawn Detection", frame)
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


def send_to_language_model(event_logs, total_blinks, gaze_outs, yawns):
    """
    Send the video analysis event logs to the Llama-based language model via Perplexity API.
    """
    # Format the logs into a readable JSON format
    formatted_logs = json.dumps(event_logs, indent=2)

    # Define the prompt for the language model
    prompt = (
        f"The following data represents events detected during a video analysis:\n\n"
        f"Total Blinks: {total_blinks}\n"
        f"Total Gaze Outside Frame: {gaze_outs}\n"
        f"Total Yawns: {yawns}\n\n"
        "Based on this data, provide a specific and detailed analysis of the person's behavior. "
        "Explain what these zero values indicate about the person's engagement, fatigue, or stress levels, "
        "and avoid discussing unrelated scenarios or generalities."
    )

    # Define the request payload
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",  # Specify the model name
        "messages": [
            {"role": "system", "content": "You are a helpful assistant analyzing behavioral data."},
            {"role": "user", "content": prompt},
        ],
    }

    # Headers including the API key for authorization
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        # Send a POST request to the API
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad HTTP responses

        # Parse the response
        response_data = response.json()
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            return "No content in response."

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return "Error while connecting to the language model."

    except json.JSONDecodeError:
        return "Error: Unable to decode the response. The response might not be in JSON format."

def analyze_video_with_language_model(video_path, max_time=60):
    """
    Process video for events and send the results to the language model for further insights.
    """
    # Run the video processing function with max_time
    results = process_video(video_path, max_time=max_time)

    if results is None:
        print("Error: Video processing failed.")
        return

    # Extract metrics
    blinks, gaze_outs, yawns, logs = results

    print(f"Total Blinks: {blinks}")
    print(f"Total Gaze Outside Frame: {gaze_outs}")
    print(f"Total Yawns: {yawns}")

    # Send event logs to the language model
    insights = send_to_language_model(logs, blinks, gaze_outs, yawns)
    if insights:
        print("\nGenerated Insights from Language Model:\n")
        print(insights)

        # Save the analysis report to the processed directory
        output_path = "data/processed/video/video_analysis_report.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
        with open(output_path, "w") as f:
            f.write(insights)
        print(f"\nVideo analysis report saved to: {output_path}")


if __name__ == "__main__":
    # Provide the correct path to the video
    video_path = "data/raw/video/SAMPLE INTERVIEW FROM THE CLIENT_720.mp4"  # Use your uploaded video

    # Analyze events and generate insights using the language model
    analyze_video_with_language_model(video_path, max_time=120)

