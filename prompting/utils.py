import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np

class llama2_platypus():
    def __init__(self, size, model_name=None):
        if model_name is None:
            if size in [7, 13, 70]:
                model_name = f"garage-bAInd/Platypus2-{size}B"
            else:
                raise Exception(f"Size {size} not available for Llama. Choose 7, 13 or 70.")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=False, add_bos_token=True)
        self.model  = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            # load_in_8bit=True,
            device_map="auto"
        )
        self.model = self.model.eval()
        self.device = self.model.device

    def prompt(self, input, question, system_context):
        # The number of tokens for the question and prompt formatting amounts to 33 tokens
        # so we can use 4064 tokens for the input text. Will use 4050 to leave some room.
        truncate_to = 4050
        input = self.tokenizer.decode(self.tokenizer.encode(input, return_tensors='pt', truncation=True, max_length=truncate_to, add_special_tokens=False)[0]) # ensure the text will fit 4096 tokens
        prompt = f"### Instruction:\n{system_context}\n\n### Input:\n{input.strip()}\n\n{question.strip()}\n\n### Response:\n"
        # ans1 = self.get_next_word_probs(prompt, allow_abstain)
        ans = self.model.generate(self.tokenizer.encode(prompt, return_tensors='pt').to(self.device), max_new_tokens=1, num_beams=1, do_sample=False)
        ans = self.tokenizer.decode(ans[0])
        ans = ans.split("### Response:")[1].strip()
        return ans