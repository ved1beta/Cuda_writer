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