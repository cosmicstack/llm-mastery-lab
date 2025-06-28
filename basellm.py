from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings

warnings.filterwarnings("ignore", message="To copy construct from a tensor.*", category=UserWarning)

class BaseLLM:
    def __init__(self, checkpoint: str, device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"):
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            cache_dir="./model_cache",
            torch_dtype=torch.float16 if device != "cpu" else torch.float32
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="./model_cache")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
    
    def _format_prompt(self, prompt: str, system: str = None):
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
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # print(f"_generate/outputs: {outputs}")
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(f"_generate/response: {response}")
        return response[0][(len(system)+len(prompt)+1):].strip()
    
    def __call__(self, *args, **kwds):
        return self._generate(*args, **kwds)

if __name__ == "__main__":
    llm = BaseLLM("microsoft/phi-3-mini-4k-instruct")
    
    # System goes here

    prompt = input("\nEnter message: \n")
    
    response = llm(prompt, system=system)
    print("\n\n")
    print(response)
    print("\n\n")