import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Chatbot:
    def __init__(self, system_prompt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.system_prompt = system_prompt

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids[0], clean_up_tokenization_spaces=False, skip_special_tokens=True)

    def generate_reply(self, prompt: str) -> str:

        appended_prompt = "\n<|user|>\n" + prompt + "\n<|end|>\n<|assistant|>\n"
        encoded_prompt = self.encode_prompt(appended_prompt)

        encoded_prompt = self.encode_prompt(appended_prompt)
        input_prompt = encoded_prompt["input_ids"].to(self.device)
        attention_mask = encoded_prompt["attention_mask"].to(self.device)

        reply = self.model.generate(input_ids=input_prompt, attention_mask=attention_mask, do_sample=True)
        new_tokens = reply[:, input_prompt.shape[1]:]

        decoded_reply = self.decode_reply(new_tokens)
        
        return decoded_reply
