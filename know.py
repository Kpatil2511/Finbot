from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "smangrul/OpenHathi-7B-Hi-v0.1-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")  # just load on CPU
    print("✅ 'Instruct' model is available locally or accessible.")
except Exception as e:
    print("❌ Instruct model not available or failed to load:\n", str(e))
