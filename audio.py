import librosa
import numpy as np
import os
import requests
import json

# Perplexity API key and endpoint
API_KEY = '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'
URL = 'https://api.perplexity.ai/chat/completions'

# Function to extract audio features
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = {
        'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1),
        'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(y).mean(),
        'rms': librosa.feature.rms(y=y).mean(),
        'tempo': librosa.beat.beat_track(y=y, sr=sr)[0]
    }
    return features

# Function to analyze voice characteristics
def analyze_voice_characteristics(features):
    insights = {}
    if features['spectral_centroid'] > 5000:
        insights['shrillness'] = 'High'
    else:
        insights['shrillness'] = 'Normal'

    if features['rms'] < 0.02 or features['spectral_bandwidth'] > 2000:
        insights['nervousness'] = 'High'
    else:
        insights['nervousness'] = 'Low'

    if features['rms'] > 0.03 and features['spectral_bandwidth'] < 1500:
        insights['confidence'] = 'High'
    else:
        insights['confidence'] = 'Low'

    return insights

# Function to extract timestamps and insights
def extract_timestamps_and_insights(audio_path, interval=5):
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    insights = []

    for start in range(0, int(duration), interval):
        start_sample = start * sr
        end_sample = min((start + interval) * sr, len(y))
        segment = y[start_sample:end_sample]

        features = {
            'mfcc': librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).mean(axis=1),
            'spectral_centroid': librosa.feature.spectral_centroid(y=segment, sr=sr).mean(),
            'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=segment, sr=sr).mean(),
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=segment, sr=sr).mean(),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(segment).mean(),
            'rms': librosa.feature.rms(y=segment).mean(),
            'tempo': librosa.beat.beat_track(y=segment, sr=sr)[0]
        }
        voice_insights = analyze_voice_characteristics(features)
        insights.append({'timestamp': f'{start}-{start + interval} seconds', 'insights': voice_insights})
    return insights

# Generate contextual insights using Perplexity API
def generate_contextual_insights_perplexity(audio_insights):
    prompt = (
        "Analyze the following data representing audio insights and provide a detailed textual report. "
        "For each timestamp, describe the speaker's shrillness, nervousness, and confidence levels, and highlight any "
        "patterns or notable changes. Avoid including any code or technical explanations."
        f"\n\nData:\n{json.dumps(audio_insights, indent=2)}"
    )

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

# Analyze a sample audio file
audio_path = "data/raw/audio/MY second interview at UPwork for 15-30 dollars per hour job__4.mp3"
interval = 5

# Extract insights
audio_insights = extract_timestamps_and_insights(audio_path, interval)
print("Audio Insights:", audio_insights)

# Generate contextual insights
contextual_insights = generate_contextual_insights_perplexity(audio_insights)
print("Contextual Insights:", contextual_insights)
