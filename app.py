from basellm import BaseLLM
from system_message import SYSTEM_MESSAGE
import gradio as gr

def generate_response(message, history):
    history.append({"role": "user", "content": message})
    llm = BaseLLM("Qwen/Qwen2.5-1.5B-Instruct")
    response = llm(message, system=SYSTEM_MESSAGE)
    return response

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=generate_response,
        type="messages",
        title="Krish-v1.1 Digital Clone",
        theme="ocean"
    )

    demo.launch()