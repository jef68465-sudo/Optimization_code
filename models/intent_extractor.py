from typing import Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class IntentExtractor(torch.nn.Module):
    def __init__(self, model_name: str, lora_config: Dict, gradient_checkpointing: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        cfg = LoraConfig(r=lora_config.get("r", 16), lora_alpha=lora_config.get("lora_alpha", 32), target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]), lora_dropout=lora_config.get("lora_dropout", 0.1))
        self.model = get_peft_model(self.model, cfg)

    def forward(self, x_inputs: Dict[str, torch.Tensor], y_inputs: Optional[Dict[str, torch.Tensor]] = None):
        if y_inputs is None:
            return self.model.generate(**x_inputs, max_length=x_inputs["input_ids"].shape[1] + 64, do_sample=True)
        input_ids = torch.cat([x_inputs["input_ids"], y_inputs["input_ids"]], dim=1)
        attention_mask = torch.cat([x_inputs["attention_mask"], y_inputs["attention_mask"]], dim=1)
        labels = input_ids.clone()
        labels[:, : x_inputs["input_ids"].shape[1]] = -100
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out

    def tokenize(self, texts, max_length: int = 512):
        return self.tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")