import streamlit as st
import subprocess
import librosa
import noisereduce as nr
import soundfile as sf
import torch
import whisper
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from scipy.signal import butter, lfilter
import os
from tempfile import NamedTemporaryFile

# Set page configuration
st.set_page_config(
    page_title="Alkimi AdCensor: AI for Adulterate Content Detection & Compliance",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check if logo files exist before displaying
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    if os.path.exists("alkimilogo.png"):
        st.image("alkimilogo.png", use_container_width=True)
    else:
        st.warning("Missing: alkimilogo.png")
with col3:
    if os.path.exists("kensaltensilogo.png"):
        st.image("kensaltensilogo.png", use_container_width=True)
    else:
        st.warning("Missing: kensaltensilogo.png")

# Title
st.title("Alkimi AdCensor: AI for Adulterate Content Detection & Compliance")

# Instruction below title
st.markdown("""
This tool extracts speech from video advertisements and detects explicit 18+ content.
It also classifies sentences as positive or negative based on sentiment analysis.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    st.write("Processing video...")

    # Convert Video to Audio (Without FFmpeg)
    def convert_video_to_mp3(input_file, output_file="audio.mp3"):
        try:
            video = VideoFileClip(input_file)
            audio = video.audio
            temp_wav = "temp_audio.wav"
            audio.write_audiofile(temp_wav, codec="pcm_s16le")
            sound = AudioSegment.from_wav(temp_wav)
            sound.export(output_file, format="mp3")
            return output_file
        except Exception as e:
            st.error(f"❌ Video-to-Audio conversion failed: {e}")
            return None

    # Audio Preprocessing (Denoising & Filtering)
    def load_audio(file_path):
        y, sr = librosa.load(file_path, sr=None)
        return y, sr

    def noise_reduction(y, sr):
        noise_sample = y[:sr]  # First 1 second as noise profile
        return nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.9)

    def bandpass_filter(y, sr, lowcut=300, highcut=3400, order=5):
        nyquist = 0.5 * sr
        low, high = lowcut / nyquist, highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, y)

    def amplify_audio(y):
        return y * 1.8  # Amplify audio by 1.8x

    def preprocess_audio(file_path, output_path="processed_audio.wav"):
        y, sr = load_audio(file_path)
        y_denoised = noise_reduction(y, sr)
        y_filtered = bandpass_filter(y_denoised, sr)
        y_amplified = amplify_audio(y_filtered)
        sf.write(output_path, y_amplified, sr)
        return output_path

    # Speech-to-Text Transcription
    def transcribe_audio(audio_file):
        model = whisper.load_model("medium")
        result = model.transcribe(audio_file)
        return result["text"]

    # 18+ Content Detection & Sentiment Analysis
    offensive_model_name = "cardiffnlp/twitter-roberta-base-offensive"
    offensive_tokenizer = AutoTokenizer.from_pretrained(offensive_model_name)
    offensive_model = AutoModelForSequenceClassification.from_pretrained(offensive_model_name)
    sentiment_pipeline = pipeline("sentiment-analysis")
    explicit_words = {"swearword1", "swearword2", "swearword3"}  # Add explicit words here

    def highlight_explicit_words(sentence):
        words = sentence.split()
        return " ".join([f'<span style="color:red; font-weight:bold;">{word}</span>' if word.lower() in explicit_words else word for word in words])

    def classify_text(text):
        inputs = offensive_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = offensive_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        labels = ["Non-Offensive", "Offensive"]
        return labels[torch.argmax(probs).item()], probs[0].tolist()

    def analyze_text(text):
        sentences = text.split('.')
        analyzed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                offensive_label, offensive_confidence = classify_text(sentence)
                sentiment = sentiment_pipeline(sentence)[0]
                highlighted_sentence = highlight_explicit_words(sentence)
                analyzed_sentences.append({
                    "sentence": highlighted_sentence,
                    "offensive_label": offensive_label,
                    "offensive_confidence": offensive_confidence,
                    "sentiment": sentiment["label"]
                })
        return analyzed_sentences

    # Processing pipeline
    audio_path = convert_video_to_mp3(temp_video_path)
    if audio_path:
        processed_audio_path = preprocess_audio(audio_path)
        transcribed_text = transcribe_audio(processed_audio_path)
        analyzed_results = analyze_text(transcribed_text)

        # Display results
        st.subheader("Analysis Results")
        for result in analyzed_results:
            st.markdown(f"**Sentence:** {result['sentence']}")
            st.markdown(f"- **Offensive Label:** {result['offensive_label']}")
            st.markdown(f"- **Sentiment:** {result['sentiment']}")
            st.write("---")












