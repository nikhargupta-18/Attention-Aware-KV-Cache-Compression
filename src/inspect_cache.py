import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
)

model.eval()

text = "Hello world. " * 20

inputs = tokenizer(
    text,
    return_tensors="pt"
)

print("Input shape:", inputs["input_ids"].shape)

with torch.no_grad():
    outputs = model(
        **inputs,
        use_cache=True,
        output_attentions=True,
    )
cache = outputs.past_key_values

print("\nCache type:")
print(type(cache))

print("\nNumber of layers:")
print(len(cache))

print("\nCache object:")
print(cache)

print("\nAttention:")
print(type(outputs.attentions))
print("Number of attention layers:", len(outputs.attentions))

for layer_idx, attention in enumerate(outputs.attentions):
    print(
        f"Layer {layer_idx}: "
        f"shape = {attention.shape}"
    )