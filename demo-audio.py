from faster_whisper import WhisperModel
import pyaudio
import wave

# ✅ Load Whisper Model (Choose "base", "small", "medium", or "large")
model = WhisperModel("small")  # "small" is fast, "large" is most accurate

# ✅ Audio Settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best with 16000 Hz
RECORD_SECONDS = 10  # Change this to adjust recording duration

p = pyaudio.PyAudio()

# ✅ Start Recording
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
frames = []

print("🎤 Recording... Speak now!")

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

print("🔴 Stopping recording...")
stream.stop_stream()
stream.close()
p.terminate()

# ✅ Save the Audio File
filename = "whisper_audio.wav"
wf = wave.open(filename, "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
print(f"Audio saved as '{filename}'.")

# ✅ Transcribe with Whisper
print("Transcribing...")
segments, info = model.transcribe("whisper_audio.wav")

print("\n📝 Transcription:")
for segment in segments:
    print(segment.text)
