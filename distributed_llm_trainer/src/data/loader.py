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