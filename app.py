import os
from basellm import BaseLLM
from system_message import SYSTEM_MESSAGE
import gradio as gr

# with open("hf_token.txt") as f:
#     hf_token = f.read().strip()

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

def generate_response(message, history):
    thread = [{"role": "system", "content": SYSTEM_MESSAGE}] + history + [{"role": "user", "content": message}]

    #TODO: Try https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
    #TODO: Try https://huggingface.co/unsloth/gemma-2b-it-bnb-4bit

    llm = BaseLLM("google/gemma-3-270m")
    response = llm(thread)
    return response

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=generate_response,
        type="messages",
        title="Krish-v1.1 Digital Clone",
        theme="ocean"
    )

    demo.launch()