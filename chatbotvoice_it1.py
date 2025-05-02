#!/usr/bin/env python3
"""
Real-time speech-to-text transcription using faster-whisper.
Uses SoundDevice for audio capture and faster-whisper for transcription.
"""

import argparse
import queue
import sys
import threading
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

class RealTimeTranscriber:
    def __init__(self, model_size="base", device="cuda", language=None):
        """Initialize the transcriber with the given model size and device."""
        print(f"Loading faster-whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.language = language
        
        # Audio parameters
        self.sample_rate = 16000  # WhisperModel expects 16kHz audio
        self.block_size = 4000    # 0.25 seconds of audio (adjust for latency vs. accuracy)
        
        # Thread-safe queue for audio blocks
        self.audio_queue = queue.Queue()
        
        # Buffer to accumulate audio for processing
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_max_size = self.sample_rate * 5  # 5 seconds max context
        
        # Thread control
        self.running = False
        self.transcription_thread = None
        
        # Track the last successful transcription time
        self.last_transcription_time = time.time()
        self.silence_threshold = 2.0  # seconds
        
        print("Transcriber initialized. Ready to capture audio.")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for the audio stream."""
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)
        
        # Add the new audio data to the queue
        self.audio_queue.put(indata.copy())

    def process_audio(self):
        """Process audio from the queue and run transcription."""
        while self.running:
            try:
                # Get audio block from queue
                audio_block = self.audio_queue.get(timeout=0.1)
                
                # Convert from stereo to mono if needed
                if audio_block.ndim > 1:
                    audio_block = audio_block.mean(axis=1)
                
                # Add to buffer
                self.audio_buffer = np.append(self.audio_buffer, audio_block)
                
                # Keep buffer at a reasonable size
                if len(self.audio_buffer) > self.buffer_max_size:
                    self.audio_buffer = self.audio_buffer[-self.buffer_max_size:]
                
                # Only process if we have enough data
                current_time = time.time()
                time_since_last_transcription = current_time - self.last_transcription_time
                
                # Process if we have enough data and either:
                # 1. It's been a while since the last transcription, or
                # 2. The queue is getting full (meaning lots of audio is coming in)
                if (len(self.audio_buffer) >= self.block_size and 
                    (time_since_last_transcription > 0.3 or self.audio_queue.qsize() > 3)):
                    
                    # Check if there's actual speech (simple energy threshold)
                    energy = np.mean(np.abs(self.audio_buffer))
                    if energy > 0.01:  # Adjust this threshold based on your microphone
                        self.transcribe_audio()
                        self.last_transcription_time = current_time
                    elif time_since_last_transcription > self.silence_threshold:
                        # If silence for too long, clear the buffer
                        self.audio_buffer = np.array([], dtype=np.float32)
                        # Print a blank line to indicate silence
                        print("\r> [silence]" + " " * 50, end="\r")
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}", file=sys.stderr)

    def transcribe_audio(self):
        """Transcribe the current audio buffer using faster-whisper."""
        try:
            # Get the current audio data (convert from float32 [-1,1] to int16)
            audio_data = (self.audio_buffer * 32767).astype(np.int16)
            
            # Convert to float32 for faster-whisper (it expects float32 in range [-1, 1])
            audio_float32 = audio_data.astype(np.float32) / 32767
            
            # Transcribe with faster-whisper
            segments, _ = self.model.transcribe(
                audio_float32, 
                language=self.language,
                beam_size=1,  # Speed up inference
                word_timestamps=False,
                suppress_blank=True,
                condition_on_previous_text=True,
                no_speech_threshold=0.6
            )
            
            # Get the segments and print the result
            text = ""
            for segment in segments:
                text += segment.text
            
            if text.strip():
                # Clear the line and print the new text
                print(f"\r> {text.strip()}" + " " * 20, end="\r")
                sys.stdout.flush()
        
        except Exception as e:
            print(f"Error in transcription: {e}", file=sys.stderr)

    def start(self):
        """Start the audio stream and transcription thread."""
        self.running = True
        
        # Start the transcription thread
        self.transcription_thread = threading.Thread(target=self.process_audio)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
        # Start the audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.block_size,
            dtype=np.float32
        ):
            print("\n=== Real-time Speech-to-Text Active ===")
            print("Speak into your microphone and see the transcription below.")
            print("Press Ctrl+C to stop.")
            print("\n> ", end="")
            
            try:
                # Keep the main thread alive until interrupted
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping transcription...")
                self.running = False
                self.transcription_thread.join(timeout=2.0)
                print("Transcription stopped.")

def main():
    parser = argparse.ArgumentParser(description="Real-time speech-to-text with faster-whisper")
    parser.add_argument(
        "--model", "-m", 
        default="large", 
        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
        help="Model size to use"
    )
    parser.add_argument(
        "--device", "-d", 
        default="cuda", 
        choices=["cpu", "cuda"], 
        help="Device to run the model on"
    )
    parser.add_argument(
        "--language", "-l", 
        default="nl", 
        help="Language code (e.g., 'en' for English, if not specified, will be auto-detected)"
    )
    
    args = parser.parse_args()
    
    transcriber = RealTimeTranscriber(
        model_size=args.model,
        device=args.device,
        language=args.language
    )
    transcriber.start()

if __name__ == "__main__":
    main()