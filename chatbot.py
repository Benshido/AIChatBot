import os
import torch
import gradio as gr
import llama_cpp
import json
from typing import List, Dict

class LocalAIChatbot:
    def __init__(self, models_directory='./models'):
        """
        Initialize the chatbot with core functionality
        
        Args:
            models_directory (str): Directory to scan for AI models
        """
        self.models_directory = models_directory
        self.current_model = None
        self.chat_history = []
        self.learning_enabled = False
        self.save_data_locally = False
        
        # Detect CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load available models
        self.available_models = self.scan_models()
    
    def scan_models(self) -> List[str]:
        """
        Scan the models directory for available models
        
        Returns:
            List of model filenames
        """
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
        
        return [
            f for f in os.listdir(self.models_directory) 
            if f.endswith(('.gguf', '.bin', '.model'))
        ]
    
    def load_model(self, model_name: str):
        """
        Load a specific AI model
        
        Args:
            model_name (str): Name of the model file to load
        """
        model_path = os.path.join(self.models_directory, model_name)
        
        try:
            # Using llama-cpp for model loading (supports various model formats)
            self.current_model = llama_cpp.Llama(
                model_path=model_path, 
                n_ctx=2048,  # Context window size
                n_gpu_layers=-1 if self.device.type == 'cuda' else 0  # Use GPU if available
            )
            print(f"Model {model_name} loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.current_model = None
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the current model
        
        Args:
            prompt (str): User's input message
        
        Returns:
            Model's generated response
        """
        if not self.current_model:
            return "No model loaded. Please select a model first."
        
        try:
            response = self.current_model(
                prompt, 
                max_tokens=250,  # Limit response length
                stop=['User:', 'Human:'],  # Stop generation at these tokens
                echo=False
            )
            generated_text = response['choices'][0]['text'].strip()
            
            # Optional: Update chat history and learning
            if self.learning_enabled:
                self.update_learning(prompt, generated_text)
            
            return generated_text
        except Exception as e:
            return f"Error generating response: {e}"
    
    def update_learning(self, prompt: str, response: str):
        """
        Update learning data if enabled
        
        Args:
            prompt (str): User's input
            response (str): Model's response
        """
        if self.save_data_locally:
            self.chat_history.append({
                'prompt': prompt,
                'response': response
            })
            
            # Optional: Save to a local JSON file
            with open('chat_history.json', 'w') as f:
                json.dump(self.chat_history, f, indent=2)
    
    def clear_chat_history(self):
        """
        Clear the current chat history
        """
        self.chat_history = []
        return []

def create_gradio_interface(chatbot):
    """
    Create a Gradio interface for the chatbot
    
    Args:
        chatbot (LocalAIChatbot): Chatbot instance
    
    Returns:
        Gradio interface
    """
    with gr.Blocks() as demo:
        # Model Selection Dropdown
        model_dropdown = gr.Dropdown(
            choices=chatbot.available_models, 
            label="Select AI Model"
        )
        model_dropdown.change(
            fn=chatbot.load_model, 
            inputs=model_dropdown
        )
        
        # Chat Interface
        chatbot_component = gr.Chatbot()
        msg = gr.Textbox(label="Enter your message")
        
        # Submit Button
        submit_btn = gr.Button("Send")
        submit_btn.click(
            fn=lambda user_msg, chat_history: (
                chatbot.generate_response(user_msg),
                chat_history + [[user_msg, chatbot.generate_response(user_msg)]]
            ),
            inputs=[msg, chatbot_component],
            outputs=[msg, chatbot_component]
        )
        
        # Clear Chat Button
        clear_btn = gr.Button("Clear Chat")
        clear_btn.click(
            fn=chatbot.clear_chat_history,
            outputs=[chatbot_component]
        )
        
        # Learning and Data Settings
        with gr.Row():
            learning_checkbox = gr.Checkbox(label="Enable AI Learning")
            learning_checkbox.change(
                fn=lambda x: setattr(chatbot, 'learning_enabled', x),
                inputs=learning_checkbox
            )
            
            save_data_checkbox = gr.Checkbox(label="Save Data Locally")
            save_data_checkbox.change(
                fn=lambda x: setattr(chatbot, 'save_data_locally', x),
                inputs=save_data_checkbox
            )
    
    return demo

def main():
    # Initialize the chatbot
    chatbot = LocalAIChatbot()
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(chatbot)
    interface.launch(share=False)

if __name__ == "__main__":
    main()