# train.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

def prepare_model_and_tokenizer(model_name="TinyLlama/TinyLlama-1.1B-intermediate"):
    """Load base model and tokenizer"""
    try:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nAvailable TinyLlama models:")
        print("- TinyLlama/TinyLlama-1.1B-intermediate")
        print("- TinyLlama/TinyLlama-1.1B-python")
        print("- TinyLlama/TinyLlama-1.1B-step-50K-105b")
        raise

def prepare_dataset(dataset_name, tokenizer, max_length=512):
    """Load and prepare the dataset"""
    try:
        print(f"Loading dataset: {dataset_name}")
        
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name)
        
        # Adjust this based on your dataset structure
        def format_medical_text(examples):
            # Modify this function based on your dataset structure
            texts = examples['text'] if 'text' in examples else examples['dialogue']
            return {'text': texts}
        
        dataset = dataset.map(format_medical_text)
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
        
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        return tokenized_dataset
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        raise

def train_model(model, tokenizer, dataset, output_dir="./medical_model"):
    """Fine-tune the model"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            save_steps=100,
            logging_steps=50,
            learning_rate=2e-5,
            fp16=True,
            gradient_checkpointing=True,
            save_total_limit=2,
            logging_dir=f"{output_dir}/logs",
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            data_collator=data_collator,
        )
        
        print("Starting training...")
        trainer.train()
        
        print(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    # Available medical datasets:
    # - "medalpaca/medical_meadow_modified" 
    # - "epfl-llm/medical-qa-zen"
    # - "martinh/medical_notes"
    
    DATASET_NAME = "medalpaca/medical_meadow_modified"
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate"  # Corrected model name
    OUTPUT_DIR = "./medical_model"
    
    try:
        # First check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU setup.")
            
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model and tokenizer
        model, tokenizer = prepare_model_and_tokenizer(MODEL_NAME)
        
        # Prepare dataset
        dataset = prepare_dataset(DATASET_NAME, tokenizer)
        
        # Train model
        train_model(model, tokenizer, dataset, OUTPUT_DIR)
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        print("\nPlease ensure:")
        print("1. You have a stable internet connection")
        print("2. You have enough GPU memory (at least 8GB recommended)")
        print("3. You're using the correct model name")
        print("4. You have access to the specified dataset")