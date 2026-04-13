from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen3-0.6B"

# 1. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Load the model on the CPU
# We use torch_dtype=torch.float32 because most CPUs handle it better than float16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": "cpu"}, 
    torch_dtype=torch.float32
)

# 3. Simple generation function
def chat_with_ai(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(chat_with_ai("Write a JSON object for a user profile."))