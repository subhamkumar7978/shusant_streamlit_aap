import streamlit as st
from pydub import AudioSegment
import os
import librosa
import soundfile as sf
import noisereduce as nr
import torch
import openai_whisper as whisper
from transformers import pipeline

def convert_video_to_audio(input_file, output_file="audio.wav"):
    try:
        video = AudioSegment.from_file(input_file)
        video.export(output_file, format="wav")
        return output_file
    except Exception as e:
        st.error(f"Error converting video to audio: {e}")
        return None

def denoise_audio(audio_file):
    try:
        audio, sr = librosa.load(audio_file, sr=None)
        reduced_noise = nr.reduce_noise(y=audio, sr=sr)
        output_file = "denoised_audio.wav"
        sf.write(output_file, reduced_noise, sr)
        return output_file
    except Exception as e:
        st.error(f"Error in noise reduction: {e}")
        return None

def transcribe_audio(audio_file):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return None

def summarize_text(text):
    try:
        summarizer = pipeline("summarization")
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return None

def main():
    st.title("AI Video-to-Text Summarizer")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        st.write("Converting video to audio...")
        audio_file = convert_video_to_audio("temp_video.mp4")
        
        if audio_file:
            st.write("Denoising audio...")
            denoised_audio = denoise_audio(audio_file)
            
            if denoised_audio:
                st.write("Transcribing audio...")
                transcript = transcribe_audio(denoised_audio)
                
                if transcript:
                    st.text_area("Transcript", transcript, height=200)
                    
                    st.write("Summarizing transcript...")
                    summary = summarize_text(transcript)
                    
                    if summary:
                        st.text_area("Summary", summary, height=100)
    
if __name__ == "__main__":
    main()













