import pyaudio
import wave
import librosa
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import time

# Initialize Audio Parameters
CHUNK = 1024  # Audio chunk size
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Sample rate

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load Vosk speech recognition model (for filler words detection)
vosk_model_path = "vosk-model-small-en-us-0.15"  # Download this model first
model = Model(vosk_model_path)
recognizer = KaldiRecognizer(model, RATE)

print("\n🎤 Speak now... (Press Ctrl+C to stop)\n")

# Start real-time processing
try:
    while True:
        frames = []
        start_time = time.time()

        # Record 5 seconds of audio
        for _ in range(int(RATE / CHUNK * 5)):  
            data = stream.read(CHUNK)
            frames.append(data)

        # Save audio to file
        wf = wave.open("temp_audio.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Analyze Confidence Features
        y, sr = librosa.load("temp_audio.wav", sr=RATE)

        # 🔹 1. Pitch Variation (Higher Variation = Nervousness)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]  
        pitch_variability = np.std(pitch_values) if len(pitch_values) > 0 else 0

        # 🔹 2. Speech Rate (Words per Second)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        speech_rate = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(y) / sr)

        # 🔹 3. Pauses & Silence Detection
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        silence_duration = sum((end - start) / sr for start, end in non_silent_intervals)

        # 🔹 4. Filler Word Detection (um, uh, aaa)
        wf = wave.open("temp_audio.wav", "rb")
        recognizer.AcceptWaveform(wf.readframes(wf.getnframes()))
        result = json.loads(recognizer.Result())
        text = result.get("text", "")
        fillers = ["uh", "um", "aaa", "like", "you know"]
        detected_fillers = [word for word in text.split() if word.lower() in fillers]

        # 🔹 Confidence Score Calculation
        confidence_score = 100 - (pitch_variability * 2 + len(detected_fillers) * 5 + silence_duration * 3)

        print(f"\n🔍 Confidence Analysis:")
        print(f"🎤 Pitch Variability: {pitch_variability:.2f} Hz")
        print(f"🗣 Speech Rate: {speech_rate:.2f} words/sec")
        print(f"⏸ Silence Duration: {silence_duration:.2f} sec")
        print(f"🤖 Filler Words: {detected_fillers}")
        print(f"💡 Confidence Score: {max(0, confidence_score):.2f}/100")

except KeyboardInterrupt:
    print("\n🔴 Stopping Audio Analysis...")
    stream.stop_stream()
    stream.close()
    p.terminate()
