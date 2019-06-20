# Main libraries to use
import torch
import torchvision

# Use CUDA if available, otherwise use CPU
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

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
valid_data = DataLoader(train_dataset, batch_size=5000, sampler=valid_sampler, num_workers=1, pin_memory=True)
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

# Task1 : Reconstruct MNIST input using Auto Encoder
# Set the dimension of a latent space to be 20 (latent_dim = 20)
Auto_Encoder = AutoEncoder(latent_dim=20).to(device)
cost_function = nn.MSELoss().to(device)    # Mean Squared Error is used
optimizer = torch.optim.Adam(Auto_Encoder.parameters()) # Adam finds its own learning rate within its algorithm

# Train our Auto Encoder
num_epoch = 20  # 1 epoch = uses all data points once
validation_criterion = 1 # criterion for validation (It will be updated to have less cost through the training)
best_network = None # model with the best validation loss will be stored

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

    if cost_valid <= validation_criterion:
        validation_criterion = cost_valid
        #torch.save(Auto_Encoder.state_dict(), './AutoEncoder_Reconstruction.pth')
        best_network = Auto_Encoder # Record the best_network with lowest validation error

# Let's test how good is our trained Auto Encoder
for data in test_data:
    input_test, label_test = data
    input_test = input_test.view(input_test.size(0), -1)
    input_test = input_test.to(device)

    # forward
    #latent_test, output_test = Auto_Encoder(input_test)
    latent_test, output_test = best_network(input_test)
    cost_test = cost_function(input_test, output_test)

    print("test loss: {:.4f}".format(cost_test.data))

# Let's plot the reconstructed images (input = input_test)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

input_test_array = np.array(input_test.cpu())
output_test_array = output_test.detach().cpu().numpy()

N = 30 # We will display N numbers in each row and each column (N x N)
       # Therefore, 450(= 30 x 15) original data and 450(= 30 x 15) reconstructed data
       # Below are the indices to be shown in the plot
       # 0      1        ...         14
       # 20     21       ...         34
       # ...    ...      ...        ...
       # 580    581      ...        594

width = 28
Recon_figure = np.zeros((width * N, width * N))

for i in range(N):
    for j in range(N):
        if j % 2 == 0:  # This is original input data
            original_temp = input_test_array[int(j/2) + N * i]
            original_temp = np.reshape(original_temp, (28,28))
            Recon_figure[i*width:(i+1)*width, j*width:(j+1)*width] = original_temp
        else:           # This is reconstructed data
            recon_temp = output_test_array[int(j/2 + N * i)]
            recon_temp = np.reshape(recon_temp, (28,28))
            Recon_figure[i*width:(i+1)*width, j*width:(j+1)*width] = recon_temp

fig1 = plt.figure(figsize = (15,15))
plt.title('Original(left) / Reconstructed(right)')
plt.imshow(Recon_figure, cmap='gray')
#plt.show()
fig1.savefig('MNIST Reconstruction')

#------------------------------------------------------------------------------------------------#

# Task2 : Visualizing latent space using matplotlib.pyplot.scatter
# Here, we set the dimension of the latent space 2 (2D plot!)
# Before plotting 2D result, we plot the previous 20D case with PCA

from sklearn.decomposition import PCA

PCA = PCA(n_components=2) # 2 Principal Components
latent_array = latent_test.detach().cpu().numpy()
PCA_array = PCA.fit_transform(latent_array)

fig2_1 = plt.figure(figsize=(10,10))
plt.scatter(PCA_array[:,0], PCA_array[:,1], c=label_test, cmap='nipy_spectral')
plt.clim(0,9)
plt.colorbar()
plt.title("A latent space with PCA (Original dim=20, Principal dim=2)")
#plt.show()
fig2_1.savefig("A latent space with PCA (dim20->dim2).png")

# Now, we see the result with Auto Encoder (latent space dimension : 2)
Auto_Encoder_2D = AutoEncoder(latent_dim=2).to(device)
cost_function_2D = cost_function    #Same as before (Mean Squared Error)
optimizer_2D = torch.optim.Adam(Auto_Encoder_2D.parameters())

# Train Session (Maybe we can apply Early Stopping)
num_epoch = 20
validation_criterion = 1
best_network_2D = None

for epoch in range(num_epoch):
    for data in train_data:
        input_train_2D, label_train_2D = data
        input_train_2D = input_train_2D.view(input_train_2D.size(0), -1)
        input_train_2D = input_train_2D.to(device)

        # forward
        latent_2D, output_2D = Auto_Encoder_2D(input_train_2D)
        cost_train_2D = cost_function_2D(input_train_2D, output_2D)

        # backward
        optimizer_2D.zero_grad()
        cost_train_2D.backward()
        optimizer_2D.step()

    print("epoch [{}/{}], train loss (2D): {:.4f}".format(epoch+1, num_epoch, cost_train_2D.data))

    # Validation
    for data in valid_data:
        input_valid_2D, label_valid_2D = data
        input_valid_2D = input_valid_2D.view(input_valid_2D.size(0), -1)
        input_valid_2D = input_valid_2D.to(device)

        output_valid_2D = Auto_Encoder_2D(input_valid_2D)[1]
        cost_valid_2D = cost_function_2D(input_valid_2D, output_valid_2D)

    print("epoch [{}/{}], validation loss (2D): {:.4f}".format(epoch+1, num_epoch, cost_valid_2D.data))

    if cost_valid_2D <= validation_criterion:
        validation_criterion = cost_valid_2D
        #torch.save(Auto_Encoder_2D.state_dict(), './AutoEncoder_Reconstruction_2D.pth')
        best_network_2D = Auto_Encoder_2D

# Test Session
for data in test_data:
    input_test_2D, label_test_2D = data
    input_test_2D = input_test_2D.view(input_test_2D.size(0), -1)
    input_test_2D = input_test_2D.to(device)

    # forward
    #latent_test_2D, output_test_2D = Auto_Encoder_2D(input_test_2D)
    latent_test_2D, output_test_2D = best_network_2D(input_test_2D)
    cost_test_2D = cost_function_2D(input_test_2D, output_test_2D)

    print("test loss (2D): {:.4f}".format(cost_test_2D.data))

latent_array_2D = latent_test_2D.detach().cpu().numpy()
output_array_2D = output_test_2D.detach().cpu().numpy()

fig2_2 = plt.figure(figsize = (10, 10))
plt.scatter(latent_array_2D[:,0], latent_array_2D[:,1], c=label_test_2D, cmap='nipy_spectral')
plt.clim(0,9)
plt.colorbar()
plt.title("A 2D latent space of a simple Auto Encoder")
#plt.show()
fig2_2.savefig("A 2D latent space.png")

#------------------------------------------------------------------------------------------------#

# Task3 : Equi-distance sample from latent space -> decode -> compare the pattern with fig2_2
# If our Auto Encoder is well-trained, then it will show a similar pattern as fig2_2

n = 30
width = 28
latent_unif = torch.empty(n*n, 2)

x_range = latent_test_2D[:,0].max() - latent_test_2D[:,0].min()
x_interval = x_range / n
y_range = latent_test_2D[:,1].max() - latent_test_2D[:,1].min()
y_interval = y_range / n

# Pick equi-distant points in the range (pick mid points rather than points in the grid)
for i in range(n):
    for j in range(n):
        latent_unif[i*n + j, 0] = latent_test_2D[:,0].min() + (i + 0.5) * x_interval
        latent_unif[i*n + j, 1] = latent_test_2D[:,1].min() + (j + 0.5) * y_interval

# Pass these equi-distance-samples data to Decoder
output_unif = Auto_Encoder_2D.decode(latent_unif.to(device))
output_unif = output_unif.reshape(n, n, width, width)

array_unif = np.zeros((width * n, width * n))

# Mind the indices! The order of equi-distant points are in the reverse order when plotting
for i in range(n):
    for j in range(n):
        unif_temp = output_unif[i,j]
        unif_temp = unif_temp.detach().cpu().numpy()
        array_unif[(n-j-1) * width:(n-j) * width, i * width : (i+1) * width] = unif_temp

fig3 = plt.figure(figsize=(30, 30))
plt.imshow(array_unif, cmap='gray')
#plt.show()
fig3.savefig('Decoded image from equi-dist data')
