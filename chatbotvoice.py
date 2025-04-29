import numpy as np
import sounddevice as sd
import threading
import time
import queue
from faster_whisper import WhisperModel

class RealtimeTranscriber:
    def __init__(self, model_size="tiny", device="cuda", compute_type="float16"):
        """
        Initialize the real-time transcriber with the specified Whisper model.
        
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large-v1, large-v2)
            device: Device to run the model on (cpu or cuda)
            compute_type: Type of compute to use (int8, float16, int8_float16)
        """
        # Audio parameters - optimized for low latency
        self.RATE = 16000  # Sample rate
        self.CHANNELS = 1  # Mono
        self.BLOCKSIZE = 512  # Smaller block size for more frequent updates
        self.SILENCE_THRESHOLD = 0.005  # Lower threshold to detect speech more easily
        self.SILENCE_DURATION = 0.8  # Shorter silence to trigger processing
        
        # Initialize Whisper model
        print(f"Loading Whisper model {model_size}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded.")
        
        # Audio queue and buffers
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.keep_running = False
        self.silence_counter = 0
        self.processing_thread = None
        self.stream = None
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for SoundDevice to capture audio chunks"""
        if status:
            print(f"Status: {status}")
            
        # Convert to float32 if not already
        audio_chunk = indata.copy()
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
            
        # Check volume level
        volume = np.abs(audio_chunk).mean()
        
        # Add chunk to buffer
        self.audio_buffer.append(audio_chunk.copy())
        
        # Check if we're in silence
        if volume < self.SILENCE_THRESHOLD:
            self.silence_counter += frames / self.RATE
            # If silence duration exceeds threshold and we have audio, process it
            if self.silence_counter >= self.SILENCE_DURATION and len(self.audio_buffer) > 0:
                # Convert buffer to processed audio
                processed_audio = np.concatenate(self.audio_buffer)
                # Convert to int16 for Whisper
                processed_audio = (processed_audio * 32767).astype(np.int16)
                # Add to processing queue
                self.audio_queue.put(processed_audio)
                # Reset buffer and silence counter
                self.audio_buffer = []
                self.silence_counter = 0
        else:
            # Reset silence counter when sound is detected
            self.silence_counter = 0

    def process_audio(self):
        """Process audio segments in the queue with Whisper"""
        while self.keep_running:
            try:
                # Get audio from queue with timeout
                audio_segment = self.audio_queue.get(timeout=0.5)
                
                try:
                    # Transcribe with Whisper - optimized parameters for speed
                    print("\nTranscribing segment...")
                    segments, _ = self.model.transcribe(
                        audio_segment, 
                        beam_size=1,  # Lower beam size for speed
                        language="en",  # Set your language for better accuracy & speed
                        vad_filter=True,  # Filter out non-speech
                        vad_parameters=dict(min_silence_duration_ms=500)  # Lower silence duration
                    )
                    
                    # Print transcription
                    for segment in segments:
                        print(f"Transcript: {segment.text}")
                except Exception as e:
                    print(f"Error during transcription: {e}")
                
                self.audio_queue.task_done()
            except queue.Empty:
                # No audio to process, continue
                continue

    def start(self):
        """Start real-time transcription"""
        self.keep_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=self.CHANNELS,
            samplerate=self.RATE,
            blocksize=self.BLOCKSIZE
        )
        self.stream.start()
        
        print("Real-time transcription started. Speak into your microphone. Press Ctrl+C to stop.")
        
        try:
            # Keep main thread alive
            while self.keep_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop transcription and clean up resources"""
        print("\nStopping transcription...")
        self.keep_running = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        if self.processing_thread:
            if self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
        
        print("Transcription stopped.")

if __name__ == "__main__":
    # Example usage - optimized for low latency
    transcriber = RealtimeTranscriber(
        model_size="tiny",  # Fastest model
        device="cuda",       # Use "cuda" if you have a compatible NVIDIA GPU
        compute_type="float16" # Use "float16" for GPU
    )
    
    try:
        transcriber.start()
    except KeyboardInterrupt:
        pass
    finally:
        transcriber.stop()