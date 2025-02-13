import torch.nn as nn
from transformers import AutoModelForCausalLM

class SimpleLLM(nn.Module):
    def __init__(self, model_name="gpt2-small"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)