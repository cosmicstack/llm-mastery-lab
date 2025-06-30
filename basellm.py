from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
from typing import overload, Union
from system_message import SYSTEM_MESSAGE 

warnings.filterwarnings("ignore", message="To copy construct from a tensor.*", category=UserWarning)

class BaseLLM:
    def __init__(self, checkpoint: str, device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"):
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
    
    @overload
    def _format_prompt(self, prompt: str, system: str = None) -> str:
        """
        Formats a single string prompt with an optional system message.
        """
    
    @overload
    def _format_prompt(self, prompt: list, system: str = None) -> str:
        """
        If the prompt is a list of dict, then does nothing
        """
    
    def _format_prompt(self, prompt: Union[str, list], system: str = None):
        if isinstance(prompt, list):
            return self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def _generate(
            self,
            prompt: str,
            system: str = None,
            max_new_tokens: int = 128,
            temperature: float = 0.7,
            top_p: float = 0.95,
    ):
        formatted_prompt = self._format_prompt(prompt, system)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()
    
    def __call__(self, *args, **kwds):
        return self._generate(*args, **kwds)

if __name__ == "__main__":
    
    # llm = BaseLLM("HuggingFaceTB/SmolLM2-360M-Instruct")
    llm = BaseLLM("Qwen/Qwen1.5-1.8B-Chat")
    
    prompt = input("\nEnter message: \n")
    
    response = llm(prompt, system=SYSTEM_MESSAGE)
    print("\n\n")
    print(response)
    print("\n\n")