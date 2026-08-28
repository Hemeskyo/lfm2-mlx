import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

base_model_path = "LiquidAI/LFM2.5-2.6B-Base"
adapter_path = "Hskyto/toolcall_adapter"
output_dir = "merged_model"

print("Loading base model in bfloat16")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.bfloat16, device_map=device
)


print("Loading adapter weights")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging adapter into base weights")
merged_model = model.merge_and_unload()

print("Saving merged model")
merged_model.save_pretrained(output_dir)

print("Saving tokenizer configuration")
tokenizer_source = (
    adapter_path
    if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json"))
    else base_model_path
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
tokenizer.save_pretrained(output_dir)

print("Merge complete!")
