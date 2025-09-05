from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model name on Hugging Face
model_name = "sarvamai/OpenHathi-7B-Chat"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model (4-bit quantized for 3060 GPU)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,        # Quantization
    device_map="auto"         # Automatically use GPU
)

# Test prompt (you can change this)
prompt = "भारत का स्टॉक मार्केट क्या है?"  # Hindi
# prompt = "मराठीमध्ये मला शेअर मार्केट समजावून सांग."  # Marathi
# prompt = "Explain the difference between stocks and bonds."  # English

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)

print("\n=== Model Response ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
