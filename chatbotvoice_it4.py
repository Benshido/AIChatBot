import sounddevice as sd
import numpy as np
import queue
import threading
import time
from pynput import keyboard
from faster_whisper import WhisperModel

# === CONFIGURATION ===
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000
CHANNELS = 1
ENERGY_THRESHOLD = 0.01
BUFFER_SECONDS = 2
LANGUAGE = "nl"

# Load the Whisper model (adjust model size as needed)
model = WhisperModel("large-v2", compute_type="float16", device="cuda")

# Shared state
audio_queue = queue.Queue()
audio_buffer = np.empty((0,), dtype=np.float32)
is_recording = False

# === Push-to-Talk Logic ===
def on_press(key):
    global is_recording
    if key == keyboard.Key.space and not is_recording:
        is_recording = True
        print("\n🎙️ Recording started...")

def on_release(key):
    global is_recording
    if key == keyboard.Key.space and is_recording:
        is_recording = False
        print("⏹️ Recording stopped.")
        return False  # Exit listener

def keyboard_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

# === Microphone callback ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ Audio status: {status}")
    if is_recording:
        audio = indata[:, 0].copy()  # mono
        audio_queue.put(audio)

# === Transcription Thread ===
def transcriber():
    global audio_buffer
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=0.1)
            audio_buffer = np.concatenate((audio_buffer, audio_chunk))

            if len(audio_buffer) >= SAMPLE_RATE * BUFFER_SECONDS:
                energy = np.mean(np.abs(audio_buffer))
                if energy > ENERGY_THRESHOLD:
                    print("🔎 Transcribing...")
                    segments, _ = model.transcribe(audio_buffer[:SAMPLE_RATE * BUFFER_SECONDS], language=LANGUAGE)
                    for seg in segments:
                        print(f"🗣️ {seg.text}")
                else:
                    print("🤫 Skipped silent segment.")
                audio_buffer = np.empty((0,), dtype=np.float32)

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break

# === Main Loop ===
def main():
    print("🟢 Hold SPACE to speak. Release to stop. Press CTRL+C to quit.")
    transcribe_thread = threading.Thread(target=transcriber, daemon=True)
    transcribe_thread.start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        blocksize=BLOCK_SIZE, dtype='float32',
                        callback=audio_callback):
        while True:
            try:
                keyboard_listener()
            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                break

if __name__ == "__main__":
    main()
