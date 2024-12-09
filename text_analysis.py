import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found. Please set it in the .env file.")

# Perplexity API endpoint
URL = 'https://api.perplexity.ai/chat/completions'

# Step 1: Load the Segmented Text
def load_segmented_text(file_path="data/processed/transcripts/segmented_dialogue.txt"):
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Failed to load segmented dialogue: {e}")
        return ""

# Step 2: Analyze the Segmented Text
def analyze_segmented_text_with_perplexity(segmented_text, model_name="llama-3.1-sonar-small-128k-online"):
    # Define the prompt
    prompt = f"""
    The following is a transcript of an interview between an interviewer and a candidate. Analyze the candidate's responses in detail. Provide insights on the following aspects:

    1. **Scenario**: What is the candidate's background and situation based on their responses?
    2. **Semantics**: Analyze the clarity, relevance, and structure of the candidate's responses.
    3. **Confidence**: Assess the confidence level of the candidate based on their tone, language, and use of filler words.
    4. **Vocabulary**: Evaluate the richness and appropriateness of the candidate's vocabulary.
    5. **Contradictions**: Identify any contradictory statements made by the candidate.
    6. **Sentiment**:
       - Assess the sentiment in the candidate's responses.
       - Does the candidate's tone suggest desperation, overconfidence, humility, or confidence?
       - Provide examples to support your analysis.

    ### Output Format:
    - **Scenario**: <Summary of the candidate's situation>
    - **Semantics**: <Detailed analysis>
    - **Confidence**: <Confidence level and examples>
    - **Vocabulary**: <Evaluation of vocabulary>
    - **Contradictions**: <Any contradictions>
    - **Sentiment**: <Analysis of tone and examples>

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
        'temperature': 0.7  # Optional, controls randomness of the output
    }

    # Define the headers with your API key
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        # Make the request to Perplexity API
        response = requests.post(URL, json=payload, headers=headers)

        if response.status_code == 200:
            # Parse the response and return the content
            output = response.json()
            return output['choices'][0]['message']['content'].strip()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        print(f"Error communicating with Perplexity API: {e}")
        return ""

# Step 3: Execute the Analysis
if __name__ == "__main__":
    # Input and output file paths
    input_file_path = "data/processed/transcripts/segmented_dialogue.txt"
    analysis_output_path = "data/processed/text_analysis/text_analysis_report.txt"

    os.makedirs(os.path.dirname(analysis_output_path), exist_ok=True)

    # Load the segmented text
    segmented_text = load_segmented_text(input_file_path)
    
    if segmented_text:
        # Analyze the text
        print("Analyzing segmented text...")
        analysis = analyze_segmented_text_with_perplexity(segmented_text)
        
        if analysis:
            # Save the analysis to a file
            with open(analysis_output_path, "w") as f:
                f.write(analysis)
            print(f"\nAnalysis saved to: {analysis_output_path}")
        else:
            print("Analysis failed. No output generated.")
    else:
        print(f"Failed to load segmented text from: {input_file_path}")
