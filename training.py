import logging
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and configuration
CONFIG = {
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Exception classes
class TrainingError(Exception):
    pass

class DataError(Exception):
    pass

# Data structures/models
class TaskDataset(Dataset):
    def __init__(self, data: pd.DataFrame, task_id: int):
        self.data = data
        self.task_id = task_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        task = self.data.iloc[idx]
        return {
            'input': torch.tensor(task['input'], dtype=torch.float32),
            'output': torch.tensor(task['output'], dtype=torch.float32),
            'task_id': torch.tensor(self.task_id, dtype=torch.long),
        }

class TaskModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(TaskModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Utility methods
def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise DataError(f'Failed to load data from {file_path}: {str(e)}')

def create_dataset(data: pd.DataFrame, task_id: int) -> TaskDataset:
    return TaskDataset(data, task_id)

def create_dataloader(dataset: TaskDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model: TaskModel, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.MSELoss):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch['input'], batch['output']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model: TaskModel, dataloader: DataLoader, loss_fn: nn.MSELoss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['input'], batch['output']
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Key functions
def train_agent(task_id: int, data_file: str):
    # Load data
    data = load_data(data_file)

    # Create dataset and dataloader
    dataset = create_dataset(data, task_id)
    dataloader = create_dataloader(dataset, CONFIG['batch_size'])

    # Create model and optimizer
    model = TaskModel(dataset.data['input'].shape[1], dataset.data['output'].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.MSELoss()

    # Train model
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        loss = train_model(model, dataloader, optimizer, loss_fn)
        end_time = time.time()
        logging.info(f'Epoch {epoch+1}, Loss: {loss:.4f}, Time: {end_time - start_time:.2f} seconds')

    # Evaluate model
    dataloader = create_dataloader(dataset, CONFIG['batch_size'])
    loss = evaluate_model(model, dataloader, loss_fn)
    logging.info(f'Final Loss: {loss:.4f}')

def main():
    # Set random seed
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])

    # Train agent
    task_id = 0
    data_file = 'data.csv'
    train_agent(task_id, data_file)

if __name__ == '__main__':
    main()