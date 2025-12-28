import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_rnacompete_data, 
    masked_mse_loss, 
    masked_spearman_correlation,
    configure_seed,
    plot
)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

class vanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(vanillaRNN, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        #RNN Layer
        self.rrn=nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0,
            nonlinearity='relu')
        
        #Fully connected layers
        self.fc1=nn.Linear(in_features=hidden_size, out_features=32)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout)
        self.fc2=nn.Linear(in_features=32, out_features=1)

    def forward(self,x):
        return x

    def train_epoch(model,loader, criterion,optimizer):
        model.train()

        for sequences, targets, masks in loader:
            sequences = sequences.to(device) 
            targets = targets.to(device)
            masks = masks.to(device)   
            optimizer.zero_grad()
            predictions= model(sequences)

    def evaluate(model,loader, criterion):


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        #LSTM Layer
        self.lstm=nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        #Fully connected layers
        self.fc1=nn.Linear(in_features=hidden_size, out_features=32)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout)
        self.fc2=nn.Linear(in_features=32, out_features=1)

    def forward(self,x):
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        #GRU Layer
        self.gru=nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        #Fully connected layers
        self.fc1=nn.Linear(in_features=hidden_size, out_features=32)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout)
        self.fc2=nn.Linear(in_features=32, out_features=1)

    def forward(self,x):

def main():
    # Hyperparameters
    batch_size = 64
    epochs = 50
    learning_rate = 0.001
    hidden_size = 64
    num_layers = 2
    dropout = 0.2

if __name__ == "__main__":
    main()