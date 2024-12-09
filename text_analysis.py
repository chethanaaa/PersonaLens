import requests
import json

# Perplexity API key
API_KEY = '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'

# Perplexity API endpoint
URL = 'https://api.perplexity.ai/chat/completions'

# Step 1: Load the Segmented Text
def load_segmented_text(file_path="segmented_dialogue.txt"):
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
    # Load the segmented text
    segmented_text = load_segmented_text("segmented_dialogue_test2.txt")
    
    if segmented_text:
        # Analyze the text
        analysis = analyze_segmented_text_with_perplexity(segmented_text)
        
        if analysis:
            # Print the analysis
            print("\nAnalysis Report:")
            print(analysis)
            
            # Save the analysis to a file
            with open("data/processed/text_analysis/text_analysis_report.txt", "w") as f:
                f.write(analysis)
            print("\nAnalysis saved to 'sentiment_analysis_report.txt'")
        else:
            print("Analysis failed. No output generated.")
