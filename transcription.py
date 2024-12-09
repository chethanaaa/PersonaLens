import whisper
import requests

# Step 1: Transcribe MP3 File
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']

API_KEY = "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"  # Replace with your Perplexity API key
URL = "https://api.perplexity.ai/v1/chat/completions"  # Ensure this endpoint is valid
    
# Step 2: Call Perplexity API to Segment Roles
def segment_roles_with_perplexity(transcript):
    
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
    {transcript}
    """

    payload = {
        'model': 'llama-3.1-sonar-small-128k-online',  # Replace with desired Perplexity model
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant specializing in data analysis.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7
    }

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(URL, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error communicating with Perplexity API: {e}")
        return None

# Step 3: Save Segmented Dialogue
def save_segmented_dialogue(segmented_dialogue, file_path="segmented_dialogue.txt"):
    try:
        with open(file_path, "w") as f:
            f.write(segmented_dialogue)
        print(f"Segmented dialogue saved to {file_path}")
    except Exception as e:
        print(f"Failed to save segmented dialogue: {e}")

# Step 4: Process Audio File
def process_audio(file_path):
    # Transcribe the audio file
    transcribed_text = transcribe_audio(file_path)
    print("Original Transcribed Text:")
    print(transcribed_text)

    # Use Perplexity to segment roles
    segmented_dialogue = segment_roles_with_perplexity(transcribed_text)

    if segmented_dialogue:
        print("\nSegmented Dialogue:")
        print(segmented_dialogue)

        # Save outputs
        save_segmented_dialogue(segmented_dialogue, "data/processed/transcripts/segmented_dialoguetest3.txt")
        save_segmented_dialogue(transcribed_text, "data/processed/transcripts/transcription_test3.txt")

    return segmented_dialogue

# Step 5: Execute
if __name__ == "__main__":
    audio_file_path = "data/raw/audio/MY second interview at UPwork for 15-30 dollars per hour job__4.mp3"  # Replace with the actual MP3 file path
    process_audio(audio_file_path)
