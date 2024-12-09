import requests
import whisper
import json
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found. Please set it in the .env file.")

# Perplexity API endpoint
URL = 'https://api.perplexity.ai/chat/completions'

# Step 1: Transcribe MP3 File
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']

def analyze_segmented_text_with_perplexity(segmented_text, model_name="llama-3.1-sonar-small-128k-online"):
    """
    Send the segmented text to the Perplexity API for analysis.
    Args:
        segmented_text (str): The input transcript to analyze.
        model_name (str): The model to use for the analysis.
    Returns:
        str: The formatted analysis response, or an empty string if an error occurs.
    """
    # Define the prompt
    prompt = f"""
    The following is a transcript of an interview. Your task is to:

    1. Label each line of dialogue with the correct speaker: "Interviewer" or "Candidate."
    2. Include every single word spoken, including filler words (e.g., "um," "uh," "like").
    3. Reformat the transcript into a clear dialogue format with speaker labels.
    4. Do not provide a summary, analysis, or commentary. Only format the dialogue as described.

    ### Output Format:
    [Interviewer]: <Text spoken by the interviewer>
    [Candidate]: <Text spoken by the candidate>

    Transcript:
    {segmented_text}
    """

    # Define the payload
    payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant specializing in data analysis.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7  # Optional: Controls the randomness of the output
    }

    # Define the headers with the API key
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        # Make the request to the Perplexity API
        response = requests.post(URL, json=payload, headers=headers)

        if response.status_code == 200:
            # Parse the response and return the formatted content
            output = response.json()
            return output.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        print(f"Error communicating with Perplexity API: {e}")
        return ""

if __name__ == "__main__":
    # Input and output file paths
    audio_file_path = "data/raw/audio/SAMPLE INTERVIEW FROM THE CLIENT_4.mp3"
    segmented_text_path = "data/processed/transcripts/segmented_dialogue.txt"
    analysis_report_path = "data/processed/transcripts/analysis_report.txt"

    os.makedirs(os.path.dirname(segmented_text_path), exist_ok=True)

    # Transcribe the audio
    segmented_text = transcribe_audio(audio_file_path)

    if segmented_text:
        # Save the segmented text
        with open(segmented_text_path, "w") as f:
            f.write(segmented_text)
        print(f"Segmented text saved to: {segmented_text_path}")

        # Analyze the text
        print("Analyzing segmented text...")
        analysis = analyze_segmented_text_with_perplexity(segmented_text)

        if analysis:
            # Save the analysis report
            with open(analysis_report_path, "w") as f:
                f.write(analysis)
            print(f"Analysis report saved to: {analysis_report_path}")
        else:
            print("Analysis failed. No output generated.")
    else:
        print("No segmented text found. Please check the input file.")
