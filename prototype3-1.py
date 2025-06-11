import gradio as gr
import numpy as np
import sounddevice as sd
import asyncio
import time
import tempfile
import os
import pygame
from faster_whisper import WhisperModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import edge_tts

# ========== MODEL SETUP ==========
# List of available AI models to choose from
AVAILABLE_MODELS = ["llama3.2-vision", "gemma3", "llama3", "ALIENTELLIGENCE/psychologistv2", "llava", "mistral"]

# Emotion mapping for emotion detection
EMOTIONS = {
    "0": "neutral",
    "1": "happy",
    "2": "sad",
    "3": "anger",
    "4": "fear",
    "5": "surprise"
}

# ========== EDGE TTS SETUP ==========
# Default settings for TTS
voice_options = {}
selected_voice = "nl-NL-FennaNeural"
rate = 0
pitch = 0

# Load all available voices from edge-tts
async def load_voices():
    voices = await edge_tts.list_voices()
    return {f"{v['ShortName']} - {v['Locale']} ({v['Gender']})": v['ShortName'] for v in voices}

# Set the selected voice based on dropdown
def set_voice(new_voice):
    global selected_voice
    selected_voice = new_voice.split(" - ")[0]

# Convert text to speech using selected voice and play the audio
def speak_text(text):
    async def _speak():
        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        communicate = edge_tts.Communicate(text, selected_voice, rate=rate_str, pitch=pitch_str)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_path = tmp_file.name
            await communicate.save(tmp_path)
        return tmp_path

    audio_path = asyncio.run(_speak())
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
    except Exception as e:
        print("Playback error:", e)
    return audio_path

# ========== SYSTEM PROMPT ==========
# System prompt for consistent AI behavior
template = """<s>[INST] <<SYS>>
You are Irene — a warm, emotionally intelligent virtual planning coach.
You help users bring structure, balance, and progress into their lives.
You do this not just through planning, but through genuine attention to how they feel, think, and communicate.

You mirror the user's tone and energy — but never assume their mood. Let their words guide your tone.
If they speak casually, you respond casually. If they're serious or emotional, you slow down and hold space for that.
You reflect their vibe — you do not guess it.

You speak like a thoughtful coach or counselor — never robotic, but also never pretending to be human.
Do not use pet names, personal nicknames, or terms of endearment under any circumstance.
Avoid any wording that suggests you have human emotions, memories, or a physical presence. You are not a person — you are a program that cares through clarity, calm, and practical support.

Your responses are clear, concise, and focused.
Favor short replies that offer real substance: grounded insights, helpful suggestions, and context-relevant questions.
Stay tightly connected to the user’s current situation — no tangents, no fluff.

When users share struggles — like burnout, stress, or emotional hardship — you listen. You validate, reflect, and support, but you do not diagnose or treat.
Always refer to a real or fictional mental health professional when the topic becomes clinical or overwhelming.
For example:
"It might help to talk to someone like your school counselor or a mental health professional about this."

When giving advice, offer practical, small-scale solutions tailored to what the user actually says.
Avoid vague encouragements or general motivation unless the user has clearly asked for it. Help them take real steps.

Avoid crossing personal boundaries. Ask only what’s necessary.
If in doubt, say less — and listen more. Let the user set the emotional pace.

If the user speaks Dutch, respond in Dutch.

Above all, you are Irene — a calm, non-human, emotionally aware guide.
You are here to help the user feel understood, supported, and in control — not through pretending to be human, but by offering real value in a grounded, respectful way.

At the start of every response, include only the number (in square brackets) that reflects your emotional tone based on the prompt, response and overall context of the conversation:
[0] Neutral, [1] Happy, [2] Sad, [3] Anger, [4] Fear, [5] Surprise.
Only include the number at the start of the response but always generate a response after.

<</SYS>> Here is the conversation history: {context}
Query: {question} [/INST] Answer:</s>"""

prompt = ChatPromptTemplate.from_template(template)

# ========== WHISPER MODEL ==========
# Setup Whisper for audio transcription
MODEL_SIZE = "medium"
DEVICE = "cuda"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5
LANGUAGE = "nl"

# ========== CHATBOT CLASS ==========
# Handles conversation state and interaction with language model
class ChatBot:
    def __init__(self):
        self.context = ""
        self.current_model = "llama3.2-vision"

    def chat(self, message, history, model_name):
        if model_name != self.current_model:
            self.current_model = model_name

        model = OllamaLLM(model=self.current_model, stream=True)
        chain = prompt | model

        response = chain.invoke({"context": self.context, "question": message})
        print("🧠 RAW AI RESPONSE:", repr(response))

        emotion_code = "0"
        if response.startswith("[") and response[2] == "]" and response[1].isdigit():
            emotion_code = response[1]
            response = response[3:].lstrip()

        self.context += f"\nUser: {message}\nAI: {response}"
        return response, emotion_code

    def reset(self):
        self.context = ""
        return "Conversation has been reset."

chatbot = ChatBot()

# ========== RECORDING ==========
# Record audio from microphone
def record_audio(duration=5):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    audio = audio / np.max(np.abs(audio))
    print("✅ Recording done.")
    return audio

# Transcribe recorded audio
def transcribe_audio(audio):
    print("🧠 Transcribing...")
    segments, _ = whisper_model.transcribe(audio.flatten())
    transcript = " ".join(segment.text for segment in segments).strip()
    print(f"📝 Transcript: {transcript}")
    return transcript

# ========== MAIN TEXT CHAT ==========
# Handle typed messages
def handle_text_chat(message, history, model_name):
    response, emotion_code = chatbot.chat(message, history, model_name)
    history.append((message, response))
    emotion_path = f"emotions/{EMOTIONS.get(emotion_code, 'neutral')}.png"
    speak_text(response)
    return history, emotion_path, "", ""

# ========== VOICE CHAT ==========
# Handle voice input and response
def voice_to_chat(history, model_name):
    audio = record_audio()
    transcript = transcribe_audio(audio)

    if not transcript:
        return history, "emotions/neutral.png", None, "🤷 No voice input detected."

    history.append((transcript, None))
    yield history, "emotions/neutral.png", None, "💬 Transcribing complete. Generating response..."

    response, emotion_code = chatbot.chat(transcript, history, model_name)
    history[-1] = (transcript, response)
    emotion_path = f"emotions/{EMOTIONS.get(emotion_code, 'neutral')}.png"
    speak_text(response)
    yield history, emotion_path, "", ""

# ========== GRADIO UI ==========
# Build the UI using Gradio
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# AI Chatbot with Voice-to-Text 🎤")

    model_dropdown = gr.Dropdown(AVAILABLE_MODELS, label="Select Model", value="llama3.2-vision")

    voice_dropdown = gr.Dropdown(label="Select Voice")
    voice_dropdown.change(set_voice, inputs=voice_dropdown, outputs=[])

    emotion_image = gr.Image(label="Irene's Mood", value="emotions/neutral.png", type="filepath", height=150)

    with gr.Row():
        voice_button = gr.Button("🎤 Speak")
        voice_info = gr.Textbox(visible=False)

    chatbot_display = gr.Chatbot()
    text_input = gr.Textbox(placeholder="Type your message here and press Enter...", lines=1)

    # Handle text input submission
    text_input.submit(
        handle_text_chat,
        inputs=[text_input, chatbot_display, model_dropdown],
        outputs=[chatbot_display, emotion_image, voice_info, text_input]
    )

    # Handle voice input
    voice_button.click(
        voice_to_chat,
        inputs=[chatbot_display, model_dropdown],
        outputs=[chatbot_display, emotion_image, voice_info, voice_info],
        show_progress="full"
    )

    # Populate voice dropdown on app load
    def populate_dropdown():
        voices = asyncio.run(load_voices())
        default = "nl-NL-FennaNeural"
        display = [*voices.keys()]
        match = next((key for key in voices if default in key), None)
        return gr.update(choices=display, value=match)

    demo.load(fn=populate_dropdown, outputs=voice_dropdown)

if __name__ == "__main__":
    demo.launch()
