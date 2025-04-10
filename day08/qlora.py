import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load a quantized model (e.g., Llama-2-7b)
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Change as needed

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token exists

# Quantized Model Loading
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,  # Use 8-bit quantization
    device_map="auto"
)

# Apply LoRA (Low-Rank Adaptation)
lora_config = LoraConfig(
    r=8,  # Rank (controls low-rank adaptation)
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.05,  # Dropout for stability
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
)

# Convert model to a PEFT model with LoRA
model = get_peft_model(model, lora_config)

# Print the number of trainable parameters
model.print_trainable_parameters()

# Load a small dataset for testing (e.g., wikitext)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")  # Small subset

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Preprocess dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qlora_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Test the fine-tuned model
input_text = "The future of artificial intelligence is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

model.eval()
with torch.no_grad():
    output = model.generate(input_ids, max_length=50)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated Text:\n", generated_text)
