import os
import shutil
from pathlib import Path
import subprocess
import sys

def create_file(path, content=""):
    """Create a file with given content."""
    with open(path, 'w') as f:
        f.write(content)

def create_project_structure():
    """Create the entire project structure with basic file contents."""
    # Project root directory
    root = Path("distributed_llm_trainer")
    if root.exists():
        choice = input(f"Directory {root} already exists. Delete it? (y/n): ")
        if choice.lower() == 'y':
            shutil.rmtree(root)
        else:
            print("Setup aborted.")
            return
    
    # Create main project directory
    root.mkdir()

    # Create directory structure
    directories = [
        "config",
        "src/data",
        "src/training",
        "src/model",
        "src/utils",
        "scripts",
        "tests",
    ]

    for dir_path in directories:
        (root / dir_path).mkdir(parents=True)
        (root / dir_path / "__init__.py").touch()

    # Create config files
    training_config = """
model:
  name: "gpt2-small"
  hidden_size: 768
  num_attention_heads: 12

training:
  batch_size: 32
  learning_rate: 0.0001
  max_epochs: 10
  gradient_clip_val: 1.0

data:
  dataset_name: "wikitext"
  subset: "wikitext-2-raw-v1"
"""
    create_file(root / "config" / "training_config.yaml", training_config)

    # Create basic files with imports
    files = {
        "src/data/loader.py": """
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_dataset(dataset_name="wikitext", subset="wikitext-2-raw-v1"):
    '''Load dataset from Hugging Face datasets'''
    return load_dataset(dataset_name, subset)

def create_dataloaders(dataset, batch_size=32):
    '''Create PyTorch DataLoaders for training'''
    train_loader = DataLoader(dataset["train"], batch_size=batch_size)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size)
    return train_loader, val_loader
""",
        "src/model/model.py": """
import torch.nn as nn
from transformers import AutoModelForCausalLM

class SimpleLLM(nn.Module):
    def __init__(self, model_name="gpt2-small"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)
""",
        "src/training/trainer.py": """
import pytorch_lightning as pl
import torch

class LLMTrainer(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input_ids"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
""",
        "requirements.txt": """
torch>=2.0.0
pytorch-lightning>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
pyyaml>=6.0
pytest>=7.3.1
""",
        "scripts/setup_environment.sh": """
#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
""",
        "scripts/run_training.py": """
import pytorch_lightning as pl
from pathlib import Path
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import get_dataset, create_dataloaders
from src.model.model import SimpleLLM
from src.training.trainer import LLMTrainer

def main():
    # Load config
    with open("config/training_config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Setup model and training
    model = SimpleLLM(config["model"]["name"])
    trainer = LLMTrainer(model, config["training"]["learning_rate"])
    
    # Setup data
    dataset = get_dataset(config["data"]["dataset_name"], config["data"]["subset"])
    train_loader, val_loader = create_dataloaders(dataset, config["training"]["batch_size"])
    
    # Train
    pl_trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
    )
    pl_trainer.fit(trainer, train_loader, val_loader)

if __name__ == "__main__":
    main()
""",
        "tests/test_basic.py": """
import pytest
from src.model.model import SimpleLLM
from src.data.loader import get_dataset

def test_model_creation():
    model = SimpleLLM()
    assert model is not None

def test_dataset_loading():
    dataset = get_dataset()
    assert dataset is not None
"""
    }

    for file_path, content in files.items():
        create_file(root / file_path, content.strip())

    # Make scripts executable
    scripts_dir = root / "scripts"
    for script in scripts_dir.glob("*.sh"):
        script.chmod(0o755)

    print("Project structure created successfully!")
    print("\nTo get started:")
    print(f"1. cd {root}")
    print("2. python -m venv venv")
    print("3. source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate")
    print("4. pip install -r requirements.txt")
    print("5. python scripts/run_training.py")

if __name__ == "__main__":
    create_project_structure()