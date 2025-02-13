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