from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

# Load the LoRA config first to get the base model name
config = PeftConfig.from_pretrained("./finetuned-stock-qa")
model_id = config.base_model_name_or_path

# Set up the same quantization config used during training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load base model with the SAME quantization as training
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./finetuned-stock-qa")

# Test inference
def ask_question(question):
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("A:")[-1].strip()

# Test your model
print("Testing the fine-tuned model...")
print(ask_question("पोर्टफोलियो क्या होता है?"))
print(ask_question("स्टॉप-लॉस ऑर्डर क्या होता है?"))
print(ask_question("स्टॉक स्प्लिट क्या होता है?"))