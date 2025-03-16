from faster_whisper import WhisperModel
import pyaudio
import wave
import os
import time

def record_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)
    wf = wave.open(file_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_chunk(model, file_path):
    segments, _ = model.transcribe(file_path)
    return " ".join(segment.text for segment in segments)

def main():
    model_size = "small.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float32")  # Try CPU first

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    accumulated_transcription = " "
    try:
        while True:
            start_time = time.time()
            chunk_file = "temp_audio.wav"
            record_chunk(p, stream=stream, file_path=chunk_file)
            record_time = time.time()
            transcription = transcribe_chunk(model, chunk_file)
            transcribe_time = time.time()
            print(transcription)
            os.remove(chunk_file)
            accumulated_transcription += transcription + " "
            print(f"Record time: {record_time - start_time}, Transcribe time: {transcribe_time - record_time}")
            time.sleep(0.1)  # Add a small delay to prevent overwhelming the system
    except KeyboardInterrupt:
        print("stopping...")
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        print("LOG: " + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()

main()