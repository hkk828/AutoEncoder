# Main libraries to use
import torch
import torchvision

# Let's load the MNIST dataset
from torchvision import transforms
from torchvision.datasets import MNIST

# Compose: a class that composes several transforms together.
img_transform = transforms.Compose([
    transforms.ToTensor(),  # converts PIL Image or numpy.ndarray in range[0,255] to a torch.FloatTensor in range[0.0, 1.0]
    transforms.Normalize((0.5, ), (0.5, ))  # apply (. - 0.5)/0.5 for each data, hence in range [-1, 1]
])
train_dataset = MNIST(root='./MNIST_data', train=True, transform=img_transform, download=True)
test_dataset = MNIST(root='./MNIST_data', train=False, transform=img_transform, download=True)

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 128 # This many data points are sampled at a time

train_sampler = SubsetRandomSampler(list(range(5000, 60000)))
valid_sampler = SubsetRandomSampler(list(range(5000)))

train_data = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1, pin_memory=True) # shuffle=True -> data reshuffled after each epoch
valid_data = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1, pin_memory=True)
test_data = DataLoader(test_dataset, batch_size=10000, num_workers=1, pin_memory=True, shuffle=True)

# Now we define our own Auto Encoder class
from torch import nn

input_size = 28 * 28    # MNIST data has size 28 x 28

# Architecture of our AutoEncoder:
# input -> 256 -> 128 -> 64 -> latent_space -> 64 -> 128 -> 256 -> output
class AutoEncoder(nn.Module):   # inherit a class "Module" from torch.nn
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim)
        )

        self.decode = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, input_size), nn.Tanh()
        )

    def forward(self, x):
        latent_var = self.encode(x)
        output = self.decode(latent_var)
        return latent_var, output

# Use CUDA if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Task1 : Reconstruct MNIST input using Auto Encoder
# Set the dimension of a latent space to be 20 (latent_dim = 20)
Auto_Encoder = AutoEncoder(latent_dim=20).to(device)
cost_function = nn.MSELoss().to(device)    # Mean Squared Error is used
optimizer = torch.optim.Adam(Auto_Encoder.parameters()) # Adam finds its own learning rate within its algorithm

# Train our Auto Encoder
num_epoch = 10  # 1 epoch = uses all data points once
validation_criterion = 1 # criterion for validation (It will be updated to have less cost through the training)

for epoch in range(num_epoch):
    for data in train_data:
        input_train, label_train = data
        input_train = input_train.view(input_train.size(0), -1)
        input_train = input_train.to(device)

        # forward
        output_train = Auto_Encoder(input_train)[1]
        cost_train = cost_function(input_train, output_train)   # We want input ~= output

        # backward
        optimizer.zero_grad()   # initialize gradients to be zero
        cost_train.backward()   # back-propagation
        optimizer.step()        # update weights using optimizer

    print("epoch [{}/{}], training loss: {:.4f}".format(epoch+1, num_epoch, cost_train.data))

    # Validation
    for data in valid_data:
        input_valid, label_valid = data
        input_valid = input_valid.view(input_valid.size(0), -1)
        input_valid = input_valid.to(device)

        # forward
        output_valid = Auto_Encoder(input_valid)[1]
        cost_valid = cost_function(input_valid, output_valid)

    print("epoch [{}/{}], validation loss: {:.4f}".format(epoch+1, num_epoch, cost_valid.data))

    if cost_valid < validation_criterion:
        validation_criterion = cost_valid
        torch.save(Auto_Encoder.state_dict(), './AutoEncoder_Reconstruction.pth')

# Let's test how good is our trained Auto Encoder
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

for data in test_data:
    input_test, label_test = data
    input_test = input_test.view(input_test.size(0), -1)

    # forward
    output_test = Auto_Encoder(input_test)[1]
    cost_test = cost_function(input_test, output_test)

    print("test loss: {.4f}".format(cost_test.data))





