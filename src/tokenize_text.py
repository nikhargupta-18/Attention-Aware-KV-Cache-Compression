from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Hello, my name is Nikhar."

tokens = tokenizer(text)

print(tokens)