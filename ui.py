import streamlit as st

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
            st.success("Files successfully uploaded. Analysis will begin shortly!")
            # Placeholder for backend integration
            st.write("**Backend integration pending.** We will connect this to the processing pipeline in the next step.")

if __name__ == "__main__":
    main()
