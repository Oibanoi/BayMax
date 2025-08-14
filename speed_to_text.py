import streamlit as st
from io import BytesIO
import os
import tempfile


from transformers import pipeline


def speed_to_text():
    transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")
    output = transcriber("test.wav")['text']
    print(output)


def speech_to_text(audio_bytes):
    # Save audio bytes to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        audio_bytes = audio_bytes.read()
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
        print("thong: ", tmp_path)
    
    # Initialize the ASR pipeline
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="vinai/PhoWhisper-small"
    )
    
    # Transcribe the audio
    result = transcriber(tmp_path)['text']

    # Clean up the temporary file
    os.unlink(tmp_path)
    
    return result


# # Title for your app
# st.title("Audio Recorder & Downloader")

# # Record audio using the microphone
# audio_bytes = st.audio_input("Speak something...", key="audio_recorder")

# if audio_bytes:
#     # Display the audio player
#     st.audio(audio_bytes, format="audio/wav")
    
#     # Create a download button
#     st.download_button(
#         label="Download Audio",
#         data=audio_bytes,
#         file_name="recorded_audio.wav",
#         mime="audio/wav"
#     )

