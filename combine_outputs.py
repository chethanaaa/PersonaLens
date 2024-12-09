import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found. Please set it in the .env file.")

# Language model API URL
BASE_URL = "https://api.perplexity.ai/chat/completions"  # Replace with your actual endpoint

def load_output(file_path):
    """
    Load textual output from individual scripts.
    Assumes outputs are saved as JSON or plain text.
    """
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return ""

def load_json_output(file_path):
    """
    Load JSON output from individual scripts.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return {}

def send_to_language_model(video_insights, audio_insights, text_insights):
    """
    Combine insights and send to the language model for final analysis.
    """
    # Combine the outputs into a single prompt
    prompt = (
        "The following are the insights from an interview analysis:\n\n"
        "### Video Analysis:\n"
        f"{video_insights}\n\n"
        "### Audio Analysis:\n"
        f"{json.dumps(audio_insights, indent=2)}\n\n"
        "### Text Analysis:\n"
        f"{text_insights}\n\n"
        "Based on these insights, generate a comprehensive final report on the interview. "
        "Highlight key observations, engagement levels, stress indicators, communication skills, "
        "and areas of improvement. Structure the report clearly and concisely."
    )

    # Define the request payload
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",  # Replace with your preferred model
        "messages": [
            {"role": "system", "content": "You are an AI assistant generating a detailed report based on interview insights."},
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

def main():
    # Run all individual scripts
    print("Running transcription.py...")
    os.system("python transcription.py")

    print("Running audio.py...")
    os.system("python audio.py")

    print("Running text_analysis.py...")
    os.system("python text_analysis.py")

    print("Running video.py...")
    os.system("python video.py")

    # Define file paths for all outputs
    video_output_path = "data/processed/video/video_analysis_report.txt"  # Video analysis output
    audio_output_path = "data/processed/audio/audio_insights.json"  # Audio insights output
    text_output_path = "data/processed/text_analysis/text_analysis_report.txt"  # Text analysis output
    final_report_path = "data/processed/final_report.txt"  # Final report path

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(final_report_path), exist_ok=True)

    # Load insights from respective outputs
    video_insights = load_output(video_output_path)
    audio_insights = load_json_output(audio_output_path)
    text_insights = load_output(text_output_path)

    # Combine the insights and generate the final report
    print("Combining insights to generate the final report...")
    final_report = send_to_language_model(video_insights, audio_insights, text_insights)

    # Save the final report to the processed directory
    with open(final_report_path, "w") as f:
        f.write(final_report)
    print(f"\nFinal Report Generated and saved to: {final_report_path}")
    print(final_report)

if __name__ == "__main__":
    main()
