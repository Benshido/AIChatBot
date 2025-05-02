import sounddevice as sd
import numpy as np
import queue
from faster_whisper import WhisperModel

# Load the model in int8 mode for faster performance
model = WhisperModel("medium", compute_type="int8_float16")

# Create an audio queue to store real-time audio chunks
audio_queue = queue.Queue()

# Audio settings
SAMPLE_RATE = 16000  # Whisper expects 16kHz
BLOCK_SIZE = 4000    # Roughly 0.25s of audio
CHANNELS = 1         # Mono input

# This callback runs whenever a new block of audio is captured
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Error: {status}")
    # Flatten stereo to mono and store in queue
    audio = indata[:, 0].copy()
    audio_queue.put(audio)

# Start recording from the microphone
with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                    blocksize=BLOCK_SIZE, callback=audio_callback):
    print("🎙️ Speak into the microphone... (Press Ctrl+C to stop)\n")

    try:
        buffer = np.empty((0,), dtype=np.float32)

        while True:
            # Read from the audio queue
            audio_chunk = audio_queue.get()
            buffer = np.concatenate((buffer, audio_chunk))

            # Once we have ~1s of audio, transcribe it
            if len(buffer) >= SAMPLE_RATE * 2:
                segment = buffer[:SAMPLE_RATE * 2]
                buffer = buffer[SAMPLE_RATE * 2:] # Keep the rest

                # Transcribe audio (in short chunk)
                segments, _ = model.transcribe(segment, language="nl")
                for seg in segments:
                    print(f"🗣️ {seg.text}")

    except KeyboardInterrupt:
        print("\n🛑 Transcription stopped.")
