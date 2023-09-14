import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np

class llama_chat_hf():
    def __init__(self, size):

        if size in [7, 13, 70]:
            model_name = f"meta-llama/Llama-2-{size}b-chat-hf"
        else:
            raise Exception(f"Size {size} not available for Llama. Choose 7, 13 or 70.")
        
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type='nf4',
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=False, add_bos_token=True)
        self.model  = AutoModelForCausalLM.from_pretrained(
            model_name,
            # quantization_config=bnb_config,
            load_in_8bit=True,
            device_map="auto"
        )
        self.model = self.model.eval()
        self.device = self.model.device
        # self.pipeline = pipeline(
        #     "text-generation",
        #     model=model_name,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        # )
        # self.device = self.pipeline.device

    def get_next_word_probs(self, prefix, allow_abstain):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prefix, return_tensors='pt').to(self.device)
            logits = self.model(input_ids).logits.squeeze()[-1]
            probabilities = torch.nn.functional.softmax(logits, dim=0)
            top_token_vals = torch.argsort(probabilities, dim=0, descending=True)
            # _, top_token_vals = torch.topk(probabilities, k=50)
            sorted_top_tokens = [self.tokenizer.decode(x).lower().strip() for x in top_token_vals]

        no_idx = sorted_top_tokens.index("no") if "no" in sorted_top_tokens else np.inf
        yes_idx = sorted_top_tokens.index("yes") if "yes" in sorted_top_tokens else np.inf

        if allow_abstain:
            ab_idx = sorted_top_tokens.index("un") if "un" in sorted_top_tokens else np.inf
            if ab_idx < yes_idx and ab_idx < no_idx:
                return "abstain"
    
        if no_idx < yes_idx:
            return "no"
        else:
            return "yes"
        # return sorted_top_tokens

    def prompt(self, user_message, allow_abstain, system_context=None):
        args = {}
        # if sampling_method == "greedy":
        #     args["num_beams"] = 1
        #     args["do_sample"] = False
        #     # args["top_k"] = None
        #     # args["top_p"] = None
        #     # args["temperature"] = None
        # elif sampling_method == "multinomial":
        #     args["num_beams"] = 1
        #     args["do_sample"] = True
        #     args["top_k"] = 10
        #     args["temperature"] = 0.1
        # elif sampling_method == "beam_search":
        #     args["num_beams"] = 3
        #     args["do_sample"] = True
        #     args["top_k"] = 10
        #     args["temperature"] = 0.1
        
        if system_context is None:
            system_context = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n<</SYS>>\n\n"
        else:
            system_context = f"[INST] <<SYS>>\n{system_context}\n<</SYS>>\n\n"

        prompt = system_context + user_message.strip() + " [/INST] "

        ans = self.get_next_word_probs(prompt, allow_abstain)
        # ans = self.model.generate(self.tokenizer.encode(prompt, return_tensors='pt').to(self.device), max_new_tokens=256)
        # ans = self.tokenizer.decode(ans[0])
        # ans = ans.split("[/INST]")[1].strip()
        return ans

class llama2_platypus():
    def __init__(self, size, model_name=None):
        if model_name is None:
            if size in [7, 13, 70]:
                model_name = f"garage-bAInd/Platypus2-{size}B"
            else:
                raise Exception("Size available for Llama. Choose 7, 13 or 70.")
        
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

    def prompt(self, user_message, system_context):
        user_message = self.tokenizer.decode(self.tokenizer.encode(user_message, return_tensors='pt', truncation=True, max_length=3800, add_special_tokens=False)[0]) # ensure the text will fit 4096 tokens
        system_context = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{system_context}"
        prompt = f"{system_context}\n\n### Input:\n{user_message.strip()}\n\n### Response:\n"
        size_prompt = len(self.tokenizer.encode(prompt, return_tensors="pt"))
        while size_prompt >= 4096:
            user_message = user_message[500:]
            prompt = f"{system_context}\n\n### Input:\n{user_message.strip()}\n\n### Response:\n"
            size_prompt = len(self.tokenizer.encode(prompt, return_tensors="pt"))

        # ans1 = self.get_next_word_probs(prompt, allow_abstain)
        ans = self.model.generate(self.tokenizer.encode(prompt, return_tensors='pt').to(self.device), max_new_tokens=1)
        ans = self.tokenizer.decode(ans[0])
        ans = ans.split("### Response:")[1].strip()
        return ans