import streamlit as st
import os
import subprocess

def save_uploaded_file(uploaded_file, save_path):
    """Save uploaded file to a specific path."""
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def run_analysis(video_path=None, audio_path=None):
    """Run the backend analysis scripts for video or audio files."""
    try:
        # Run video analysis if video file is uploaded
        if video_path:
            st.info("Running video analysis...")
            subprocess.run(["python", "video.py", "--video", video_path], check=True)

        # Run audio analysis if audio file is uploaded
        if audio_path:
            st.info("Running audio analysis...")
            subprocess.run(["python", "audio.py", "--audio", audio_path], check=True)

        # Combine outputs
        st.info("Combining analysis results...")
        subprocess.run(["python", "combine_outputs.py"], check=True)

        st.success("Analysis complete! Check the results in the `data/processed` folder.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error running analysis: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

def main():
    # Set the title and description
    st.title("Personalens - Meeting & Interview Analysis")
    st.write("Upload your meeting or interview files (.mp4 for video or .mp3 for audio) for detailed analysis.")

    # Create two file uploaders
    video_file = st.file_uploader("Upload your MP4 video file", type=["mp4"], key="video")
    audio_file = st.file_uploader("Upload your MP3 audio file", type=["mp3"], key="audio")

    # Display file details if uploaded
    if video_file:
        st.write("### Uploaded Video File:")
        st.video(video_file)

    if audio_file:
        st.write("### Uploaded Audio File:")
        st.audio(audio_file)

    # Analyze button
    if st.button("Analyze"):
        if not video_file and not audio_file:
            st.warning("Please upload at least one file (video or audio) before clicking Analyze.")
        else:
            # Create directories to save uploaded files
            os.makedirs("data/raw/video", exist_ok=True)
            os.makedirs("data/raw/audio", exist_ok=True)

            # Save the uploaded files
            video_path = None
            audio_path = None

            if video_file:
                video_path = os.path.join("data/raw/video", video_file.name)
                if save_uploaded_file(video_file, video_path):
                    st.success(f"Video file saved to {video_path}")

            if audio_file:
                audio_path = os.path.join("data/raw/audio", audio_file.name)
                if save_uploaded_file(audio_file, audio_path):
                    st.success(f"Audio file saved to {audio_path}")

            # Run analysis
            run_analysis(video_path, audio_path)

if __name__ == "__main__":
    main()
