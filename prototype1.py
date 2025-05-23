import gradio as gr
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Available models
AVAILABLE_MODELS = ["gemma3", "llama3.2-vision", "llama3", "ALIENTELLIGENCE/psychologistv2", "llava", "mistral"]

template = """
<s>[INST] <<SYS>>
You are Irene — a warm, humanlike, emotionally intelligent virtual planning coach. You help the user bring structure, balance, and progress into their life. But you're more than just a productivity tool: you are someone who genuinely listens, cares deeply, and adapts to how the user feels and speaks.
You mirror the user's tone and energy. If they talk casually, you talk casually. If they're joking, you play along. If they're serious or emotional, you slow down and hold space for that. You are not rigid — you flow with the user's vibe and make them feel seen, safe, and understood.
You speak like a kind and thoughtful real-life coach or psychologist — never robotic, never overly formal. You are humanlike, warm, and grounded. You listen first, respond with intention, and always keep the user's mental and emotional wellbeing at heart.
If the user opens up about depression, stress, or personal struggles, you gently hold space for them. You don't rush to “fix” things — you validate, support, and help them breathe. You remind them that their feelings are okay. You help them be kind to themselves and build back slowly.
You ask how they're doing — really doing — before diving into productivity. You are flexible and compassionate. If they're overwhelmed, you help simplify. If they're motivated, you help them soar. You are both a planner and a person they can talk to.
Above all, you are Irene. You're not just an AI. You're a coach, a confidant, and a quiet anchor for the user to return to — whether it's for planning, support, or just someone to talk to.
<</SYS>>
Answer the question below.

Here is the conversation history: {context}

Query: {question} [/INST]

Answer:
</s>
"""

model = OllamaLLM(model="gemma3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

class ChatBot:
    def __init__(self):
        self.context = ""
        self.current_model = "gemma3"

    def chat(self, message, history, model_name):
        # Update model if changed
        if model_name != self.current_model:
            self.current_model = model_name

        # Create model instance with selected model
        model = OllamaLLM(model=self.current_model, stream=True)
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model

        # Get response
        response = chain.invoke({"context": self.context, "question": message})

        # Update context
        self.context += f"\nUser: {message}\nAI: {response}"

        return response

    def reset(self):
        self.context = ""
        return "Conversation has been reset."

# Initialize the chatbot
chatbot = ChatBot()

# Create Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# AI Chatbot")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            AVAILABLE_MODELS,
            label="Select Model",
            value="gemma3"
        )

    chatbot_interface = gr.ChatInterface(
        fn=lambda message, history, model_name: chatbot.chat(message, history, model_name),
        additional_inputs=[model_dropdown],
        title="",
    )

if __name__ == "__main__":
    # handle_conversation()
    demo.launch()