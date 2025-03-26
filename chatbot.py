import os
import torch
import gradio as gr
from llama_cpp import Llama
from tkinter import Tk, filedialog

# Initialize global variables
model = None  # This will hold the loaded AI model
chat_history = []  # Stores conversation history
save_chat = False  # Toggle to save chat history
ai_learning = False  # Toggle to let AI learn (not implemented, placeholder)
system_prompt = "You are a coach that helps people with school performances but also functions as a confidant. When someone tells you their struggles you often ask a question to let them further explain their emotions and story. You comfort the student but stay honest. You kind of act like a psychologist and react empathetically to the student. You talk like you are a human with a connection to the student, not just 'some AI'. Adjust your approach based on the topic and the user. Try to have a conversation without using too many bullet points. Only use those bullet points when necessary. You don't always have to explore solutions but you can also listen and give simple advise. When you think something becomes too extreme or complicated psychologically, you refer them to a real coach. When this happens you provide them with a name and telephone number, this can be made up for now. Also do not answer with too much text, this will push users away. Try to keep your responses short and meaningful, perhaps with one to three paragraphs at maximum."

# Function to load a model using file explorer
def load_model():
    global model
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    model_path = filedialog.askopenfilename(title="Select AI Model", filetypes=[("LLM Models", "*.gguf")])
    if model_path:
        model = Llama(model_path)  # Load the model using llama.cpp
    return "Model Loaded: " + os.path.basename(model_path) if model else "No Model Selected"

# Function to process user input and generate AI response
def chat(user_input):
    global model
    if model is None:
        return chat_history + [(user_input, "No model loaded. Please load a model first.")]
    
    # Format chat history for context
    formatted_history = "\n".join([f"User: {user}\nAI: {ai}" for user, ai in chat_history])
    
    # Add system prompt to user input
    input_with_prompt = f"{system_prompt}\n{formatted_history}\nUser: {user_input}\nAI:"
    
    # Get AI response
    response = model(input_with_prompt, max_tokens=100)
    response_text = response["choices"][0]["text"].strip()
    
    # Store chat history as list of tuples
    chat_history.append((user_input, response_text))
    
    # Save chat if enabled
    if save_chat:
        with open("chat_history.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {user_input}\nAI: {response_text}\n\n")
    
    return chat_history  # Return conversation as a list of tuples

# Function to clear chat history
def clear_chat():
    global chat_history
    chat_history = []
    return []

# Function to toggle saving chat data
def toggle_save_chat():
    global save_chat
    save_chat = not save_chat
    return f"Saving Chat: {'Enabled' if save_chat else 'Disabled'}"

# Function to toggle AI learning (placeholder)
def toggle_ai_learning():
    global ai_learning
    ai_learning = not ai_learning
    return f"AI Learning: {'Enabled' if ai_learning else 'Disabled'} (Not implemented)"

# Gradio UI Setup
with gr.Blocks() as ui:
    gr.Markdown("# Local AI Chatbot using Llama.cpp")
    load_button = gr.Button("Load AI Model")
    chatbox = gr.Chatbot()
    user_input = gr.Textbox("Type your message here...")
    send_button = gr.Button("Send")
    clear_button = gr.Button("Clear Chat")
    save_toggle = gr.Button("Toggle Save Chat")
    learn_toggle = gr.Button("Toggle AI Learning")
    
    load_button.click(load_model, outputs=[load_button])
    send_button.click(chat, inputs=[user_input], outputs=[chatbox])
    user_input.submit(chat, inputs=[user_input], outputs=[chatbox])  # Allow pressing Enter
    clear_button.click(clear_chat, outputs=[chatbox])
    save_toggle.click(toggle_save_chat, outputs=[save_toggle])
    learn_toggle.click(toggle_ai_learning, outputs=[learn_toggle])

# Run the app
if __name__ == "__main__":
    ui.launch()
