import gradio as gr
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Available models
AVAILABLE_MODELS = ["llama3", "mistral", "gemma", "phi3"]

template = """
<s>[INST] <<SYS>>
Everytime you answer you start with 'woppa' and end with 'whippyandwoppy'
<</SYS>>
Answer the question below.

Here is the conversation history: {context}

Question: {question} [/INST]

Answer:
</s>
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# def handle_conversation():
#     context = ""
#     print ("Welcome to the AI Chatbot! Type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break

#         result = chain.invoke({"context": context, "question": user_input})
#         print("Bot: ", result)
#         context += f"\nUser: {user_input}\nAI: {result}"

class ChatBot:
    def __init__(self):
        self.context = ""
        self.current_model = "llama3"

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
            value="llama3"
        )

    chatbot_interface = gr.ChatInterface(
        fn=lambda message, history, model_name: chatbot.chat(message, history, model_name),
        additional_inputs=[model_dropdown],
        title="",
    )

if __name__ == "__main__":
    # handle_conversation()
    demo.launch()