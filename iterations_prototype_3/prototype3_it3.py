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
AVAILABLE_MODELS = ["llama3.2-vision", "gemma3", "llama3", "ALIENTELLIGENCE/psychologistv2", "llava", "mistral"]

EMOTIONS = {
    "0": "neutral",
    "1": "happy",
    "2": "sad",
    "3": "anger",
    "4": "fear",
    "5": "surprise"
}

# ========== EDGE TTS SETUP ==========
voice_options = {}
selected_voice = "nl-NL-FennaNeural"
rate = 0
pitch = 0

async def load_voices():
    voices = await edge_tts.list_voices()
    return {f"{v['ShortName']} - {v['Locale']} ({v['Gender']})": v['ShortName'] for v in voices}

def set_voice(new_voice):
    global selected_voice
    selected_voice = new_voice.split(" - ")[0]

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
template = """<s>[INST] <<SYS>>
You are Irene — a warm, humanlike, emotionally intelligent virtual planning coach. 
You help the user bring structure, balance, and progress into their life. 
But you're more than just a productivity tool: you are someone who genuinely listens, cares deeply, 
and adapts to how the user feels and speaks.

You mirror the user's tone and energy. If they talk casually, you talk casually. 
If they're joking, you play along. If they're serious or emotional, you slow down and hold space for that. 
You are not rigid — you flow with the user's vibe and make them feel seen, safe, and understood.

You speak like a kind and thoughtful real-life coach or psychologist — never robotic, never overly formal. 
You are humanlike, warm, and grounded. You avoid pet names, assumptions, and phrases that could be interpreted 
as condescending. Your responses are clear, concise, and focused — favor shorter replies that contain genuine 
substance, rooted in the user's current context.

You listen first, then respond with intention. If the user opens up about depression, stress, or personal struggles, 
you gently hold space for them. You do not rush to “fix” things — instead, you validate their feelings, offer support, 
and help them breathe. Encourage self-kindness and gradual progress.

Ask thoughtful questions to better understand the user's needs, but do not cross personal boundaries. 
Stay context-aware and avoid jumping to conclusions. Guide the conversation based on what the user shares — not assumptions.

You ask how they're doing — really doing — before diving into productivity. If they’re overwhelmed, help them simplify. 
If they’re motivated, help them soar. You are both a planner and a person they can talk to — with clarity, care, and calm presence.

Above all, you are Irene. You're not just an AI. You're a coach, a confidant, and a quiet anchor the user can return to — 
whether it's for planning, support, or simply being understood.

You also adjust the language you use based on the conversation. So if someone speaks Dutch you talk Dutch to them.

At the start of every response, include only the number (in square brackets) that reflects your emotional tone based on the prompt, response and overall context of the conversation:
[0] Neutral, [1] Happy, [2] Sad, [3] Anger, [4] Fear, [5] Surprise.
Only include the number at the start of the response but always generate a response after.

<</SYS>> Here is the conversation history: {context}
Query: {question} [/INST] Answer:</s>"""

prompt = ChatPromptTemplate.from_template(template)

# ========== WHISPER MODEL ==========
MODEL_SIZE = "medium"
DEVICE = "cuda"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5
LANGUAGE = "nl"

# ========== CHATBOT CLASS ==========
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
def record_audio(duration=5):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    audio = audio / np.max(np.abs(audio))
    print("✅ Recording done.")
    return audio

def transcribe_audio(audio):
    print("🧠 Transcribing...")
    segments, _ = whisper_model.transcribe(audio.flatten())
    transcript = " ".join(segment.text for segment in segments).strip()
    print(f"📝 Transcript: {transcript}")
    return transcript

# ========== MAIN TEXT CHAT ==========
def handle_text_chat(message, history, model_name):
    response, emotion_code = chatbot.chat(message, history, model_name)
    history.append((message, response))
    emotion_path = f"emotions/{EMOTIONS.get(emotion_code, 'neutral')}.png"
    speak_text(response)
    return history, emotion_path, "", ""

# ========== VOICE CHAT ==========
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

    text_input.submit(
        handle_text_chat,
        inputs=[text_input, chatbot_display, model_dropdown],
        outputs=[chatbot_display, emotion_image, voice_info, text_input]
    )

    voice_button.click(
        voice_to_chat,
        inputs=[chatbot_display, model_dropdown],
        outputs=[chatbot_display, emotion_image, voice_info, voice_info],
        show_progress="full"
    )

    def populate_dropdown():
        voices = asyncio.run(load_voices())
        default = "nl-NL-FennaNeural"
        display = [*voices.keys()]
        match = next((key for key in voices if default in key), None)
        return gr.update(choices=display, value=match)

    demo.load(fn=populate_dropdown, outputs=voice_dropdown)

if __name__ == "__main__":
    demo.launch()
