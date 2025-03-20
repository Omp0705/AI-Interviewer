import pyaudio
import wave


# Audio Settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
OUTPUT_FILENAME = "recorded_audio.wav"

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

frames = []
print("🎤 Recording... Press Ctrl+C to stop.")

try:
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
except KeyboardInterrupt:
    print("\n🔴 Stopping recording...")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio
with wave.open(OUTPUT_FILENAME, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))

print(f"✅ Recording saved as {OUTPUT_FILENAME}")
