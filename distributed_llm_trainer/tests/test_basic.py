import pytest
from src.model.model import SimpleLLM
from src.data.loader import get_dataset

def test_model_creation():
    model = SimpleLLM()
    assert model is not None

def test_dataset_loading():
    dataset = get_dataset()
    assert dataset is not None