# -*- coding: utf-8 -*-
#https://github.com/MedMNIST/MedMNIST
# Deep Learning Homework 2 - Question 1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST, INFO
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data Loading
data_flag = 'bloodmnist'
print(data_flag)
info = INFO[data_flag]
print(len(info['label']))
n_classes = len(info['label'])

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Hyperparameters
batch_size=64
epochs=200
learning_rate=0.001

# initialize the model
# get an optimizer
# get a loss criterion
### YOUR CODE HERE ###
class CNN(nn.Module):

    def __init__(self, maxpool=False, use_softmax=False):
        super(CNN,self).__init__()
        self.maxpool = maxpool
        self.use_softmax = use_softmax
        num_classes=8

        # Convolution Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1)

        # Implementation for Q2.2
        if maxpool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1_input_size = 128 * 3 * 3
        else:
            self.fc1_input_size=128*28*28

        #Fully connected layers
        self.fc1 = nn.Linear(in_features=self.fc1_input_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self,x):
        #1)convolution and relu layers + conditional maxpooling
        x=F.relu(self.conv1(x))
        if self.maxpool:
            x=self.pool(x)
        
        #2)convolution and relu layers + conditional maxpooling
        x=F.relu(self.conv2(x))
        if self.maxpool:
            x=self.pool(x)

        #3)convolution and relu layers + conditional maxpooling
        x=F.relu(self.conv3(x))
        if self.maxpool:
            x=self.pool(x)

        #Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer 1 + relu
        x=F.relu(self.fc1(x))

        # Fully connected layer 2
        x=self.fc2(x)

        #conditional softmax
        if self.use_softmax:
            x=F.softmax(x, dim=1)

        return x

#Training Function
def train_epoch(loader, model, criterion, optimizer):
    ### YOUR CODE HERE ###
    model.train()
    total_loss=0.0
    for imgs, labels in loader:
        imgs=imgs.to(device)
        labels=labels.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs=model(imgs)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    return total_loss / len(loader)

#Evaluation Function
def evaluate(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)

#Plotting function
def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')

def main(use_maxpool=False, use_softmax=False):
    #Load Data
    train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
    val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
    test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --------- Before Training ----------
    total_start = time.time()

    #Initialize model, criterion, optimizer
    model = CNN(maxpool=use_maxpool, use_softmax=use_softmax).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # training loop
    ### you can use the code below or implement your own loop ###
    train_losses = []
    val_accs = []
    test_accs = []
    for epoch in range(epochs):

        epoch_start = time.time()

        train_loss = train_epoch(train_loader, model, criterion, optimizer)
        val_acc = evaluate(val_loader, model)
        test_acc = evaluate(test_loader, model)

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        print(f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Time: {epoch_time:.2f} sec")

    #Test Accuracy
    # test_acc = evaluate(test_loader, model)
    # print("Test Accuracy:", test_acc)
    # test_accs.append(test_acc)

    config = f"maxpool{use_maxpool}_softmax{use_softmax}"

    #Save the model
    torch.save(model.state_dict(), "bloodmnist_cnn_{config}.pth")
    print("Model saved as bloodmnist_cnn_{config}.pth")

    # --------- After Training ----------
    total_end = time.time()
    total_time = total_end - total_start

    print(f"\nTotal training time: {total_time/60:.2f} minutes "
        f"({total_time:.2f} seconds)")

    epochs_list = list(range(1, epochs+1))

    plot(epochs_list, train_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs_list, val_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    plot(epochs_list, test_accs, ylabel='Accuracy', name='CNN-test-accuracy-{}'.format(config))

    return {
        'train_losses': train_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'total_time': total_time,
    }

if __name__ == '__main__':
    results = main(use_maxpool=False, use_softmax=False)