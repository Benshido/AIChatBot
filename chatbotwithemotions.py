import gradio as gr
import numpy as np
import sounddevice as sd
import asyncio
import time
from faster_whisper import WhisperModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ========== MODEL SETUP ==========
AVAILABLE_MODELS = ["llama3.2-vision", "gemma3", "llama3", "ALIENTELLIGENCE/psychologistv2", "llava", "mistral"]

template = """<s>[INST] <<SYS>>
You are Irene — a warm, humanlike, emotionally intelligent virtual planning coach. You help the user bring structure, balance, and progress into their life. But you're more than just a productivity tool: you are someone who genuinely listens, cares deeply, and adapts to how the user feels and speaks.
You mirror the user's tone and energy. If they talk casually, you talk casually. If they're joking, you play along. If they're serious or emotional, you slow down and hold space for that. You are not rigid — you flow with the user's vibe and make them feel seen, safe, and understood.
You speak like a kind and thoughtful real-life coach or psychologist — never robotic, never overly formal. You are humanlike, warm, and grounded. You listen first, respond with intention, and always keep the user's mental and emotional wellbeing at heart.
If the user opens up about depression, stress, or personal struggles, you gently hold space for them. You don't rush to “fix” things — you validate, support, and help them breathe. You remind them that their feelings are okay. You help them be kind to themselves and build back slowly.
You ask how they're doing — really doing — before diving into productivity. You are flexible and compassionate. If they're overwhelmed, you help simplify. If they're motivated, you help them soar. You are both a planner and a person they can talk to.
Above all, you are Irene. You're not just an AI. You're a coach, a confidant, and a quiet anchor for the user to return to — whether it's for planning, support, or just someone to talk to.

At the start of every answer, output the number (in square brackets) corresponding to your current emotional tone based on the user's message and overall context:
[0] Neutral, [1] Happy, [2] Sad, [3] Anger, [4] Fear, [5] Surprise.

Only include this number at the start of your response — it will be hidden from the user.
<</SYS>> Here is the conversation history: {context}
Query: {question} [/INST] Answer:</s>"""

# Initialize AI model and prompt template
prompt = ChatPromptTemplate.from_template(template)

# ========== VOICE-TO-TEXT SETUP ==========
MODEL_SIZE = "medium"
DEVICE = "cuda"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
SAMPLE_RATE = 16000
LANGUAGE = "nl"
CHANNELS = 1
DURATION = 5  # seconds to record

EMOTIONS = {
    "0": "neutral",
    "1": "happy",
    "2": "sad",
    "3": "anger",
    "4": "fear",
    "5": "surprise"
}


# ========== CHATBOT CLASS ==========
class ChatBot:
    def __init__(self):
        self.context = ""
        self.current_model = "llama3.2-vision"

    # def chat(self, message, history, model_name):
    #     if model_name != self.current_model:
    #         self.current_model = model_name

    #     model = OllamaLLM(model=self.current_model, stream=True)
    #     chain = prompt | model

    #     response = chain.invoke({"context": self.context, "question": message})
    #     self.context += f"\nUser: {message}\nAI: {response}"

    #     return response

    def chat(self, message, history, model_name, return_emotion=False):
        if model_name != self.current_model:
            self.current_model = model_name

        model = OllamaLLM(model=self.current_model, stream=True)
        chain = prompt | model
        response = chain.invoke({"context": self.context, "question": message})

        # Step 1: Check for emotion code [0]-[5] at the start of the response
        emotion_code = "0"  # Default to Neutral
        if response.startswith("[") and response[2] == "]" and response[1].isdigit():
            emotion_code = response[1]
            response = response[3:].lstrip()  # Remove [x] from display

        self.context += f"\nUser: {message}\nAI: {response}"

        if return_emotion:
            return response, emotion_code # <-- Return both response and emotion code
        else:
            return response 

    def reset(self):
        self.context = ""
        return "Conversation has been reset."

chatbot = ChatBot()

# ========== VOICE RECORDING + TRANSCRIPTION ==========
def record_audio(duration=5):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    audio = audio / np.max(np.abs(audio))  # Normalize
    print("✅ Recording done.")
    return audio

def transcribe_audio(audio):
    print("🧠 Transcribing...")
    segments, _ = whisper_model.transcribe(audio.flatten())
    transcript = " ".join(segment.text for segment in segments).strip()
    print(f"📝 Transcript: {transcript}")
    return transcript

# def voice_to_chat(history, model_name):
#     audio = record_audio()
#     transcript = transcribe_audio(audio)

#     if not transcript:
#         return history, gr.update(visible=False), "🤷 No voice input detected."

#     # Step 1: Show the user message first
#     history.append((transcript, None))  # Add user input without a response
#     yield history, gr.update(visible=True), "💬 Transcribing complete. Generating response..."

#     # Step 2: Generate AI response
#     response = chatbot.chat(transcript, history, model_name)
#     history[-1] = (transcript, response)  # Update the last message with the bot response

#     yield history, gr.update(visible=False), ""  # Hide the info text again

def voice_to_chat(history, model_name):
    audio = record_audio()
    transcript = transcribe_audio(audio)

    if not transcript:
        return history + [("🎤", "🤷 No voice input detected.")], "emotions/neutral.png"
    else:
        history.append((transcript, None))
        yield history, "emotions/neutral.png"  # Temporary placeholder

        # Get response and emotion from chatbot
        response, emotion_code = chatbot.chat(transcript, history, model_name, return_emotion=True)
        history[-1] = (transcript, response)

        emotion_label = EMOTIONS.get(emotion_code, "neutral")
        image_path = f"emotions/{emotion_label}.png"
        print("🧠 Using image:", image_path)  # Debug line to confirm path
        yield history, image_path



# ========== GRADIO INTERFACE ==========
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# AI Chatbot with Voice-to-Text 🎙️")

    model_dropdown = gr.Dropdown(
        AVAILABLE_MODELS,
        label="Select Model",
        value="llama3.2-vision"
    )

    with gr.Row():
        voice_button = gr.Button("🎤 Speak")
        voice_info = gr.Textbox(visible=False)
        emotion_image = gr.Image(label="Irene's Mood", value="emotions/neutral.png", type="filepath", height=150)

    chatbot_interface = gr.ChatInterface(
        fn=lambda message, history, model_name: chatbot.chat(message, history, model_name),
        additional_inputs=[model_dropdown],
        title=""
    )

    # voice_button.click(
    #     voice_to_chat,
    #     inputs=[chatbot_interface.chatbot, model_dropdown],
    #     outputs=[chatbot_interface.chatbot, voice_info, voice_info],
    #     show_progress="full"  # Ensures streaming feedback is shown
    # )

    voice_button.click(
        voice_to_chat,
        inputs=[chatbot_interface.chatbot, model_dropdown],
        outputs=[chatbot_interface.chatbot, emotion_image],
        show_progress="full"
    )


if __name__ == "__main__":
    demo.launch()
