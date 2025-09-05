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

def ask_question(question, conversation_history=None):
    """
    Generate response with optional conversation history
    """
    if conversation_history:
        # Build conversation context
        prompt = conversation_history + f"\nQ: {question}\nA:"
    else:
        prompt = f"Q: {question}\nA:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new answer
    if "A:" in response:
        answer = response.split("A:")[-1].strip()
    else:
        answer = response.replace(prompt, "").strip()
    
    return answer

def interactive_chat():
    """
    Start an interactive chat session with the model
    """
    print("🤖 Welcome to Stock Chatbot! (Type 'quit' to exit, 'clear' to reset conversation)")
    print("-" * 50)
    
    conversation_history = ""
    turn_count = 0
    
    while True:
        # Get user input
        user_input = input("\n👤 You: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("🤖 Goodbye!")
            break
            
        if user_input.lower() in ['clear', 'reset']:
            conversation_history = ""
            turn_count = 0
            print("🤖 Conversation history cleared!")
            continue
            
        if not user_input:
            continue
            
        # Generate response
        print("🤖 Thinking...", end=" ", flush=True)
        try:
            response = ask_question(user_input, conversation_history)
            print(f"\r🤖 Bot: {response}")
            
            # Update conversation history (keep last 3 turns to avoid too long context)
            if turn_count < 3:
                conversation_history += f"\nQ: {user_input}\nA: {response}"
                turn_count += 1
            else:
                # Keep only the most recent exchange
                lines = conversation_history.split('\n')
                conversation_history = '\n'.join(lines[-6:]) + f"\nQ: {user_input}\nA: {response}"
                
        except Exception as e:
            print(f"\r❌ Error: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    interactive_chat()