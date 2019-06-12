import torch
import torchvision
from torch import nn
from torchvision import transforms

# Load MNIST data
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

img_transform = transforms.Compose([
    transforms.ToTensor(),   # converts PIL.Image.Image into a tensor
    transforms.Normalize((0.5,), (0.5,)) #For a single channel, mean/std for each single element
])
train_dataset = MNIST('./data/', transform = img_transform, train=True, download=True)
test_dataset = MNIST('./data/', transform = img_transform, train=False, download=True)
#validation_dataset = MNIST.validation

batch_size = 128
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=10000, shuffle= True)

input_size = 28 * 28
#define Auto Encoder class
class autoencoder(nn.Module):
    # input -> 256 -> 128 -> 64 -> latent_space ->64 -> 128 -> 256 -> output(=input)
    def __init__(self, latent_dim):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(True),
            nn.Linear(256, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, latent_dim)) # Activation function?

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True),
            nn.Linear(256, input_size), nn.Tanh())

    def forward(self, x):
        x_latent = self.encoder(x)
        x_output = self.decoder(x_latent)
        return x_latent, x_output

# latent space dimension = 20
model = autoencoder(latent_dim=20)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters()) #Learning rate and weight decay are not necessary -> Over-Tuning if used. This is why Adam is good!

#Training Auto Encoder using MNIST training data
num_epochs = 10
validation_loss = 1 #initialized to 100, but will be updated if less loss recorded.
for epoch in range(num_epochs):
    for data in train_data:
        inputs, label = data
        inputs = inputs.view(inputs.size(0), -1) # view changes size, inputs.size(0) = 128 -> converts size of 'inputs' to [128, 784]

        # forward
        _, output_train = model(x=inputs)
        train_loss = criterion(output_train, inputs)

        # backward
        optimizer.zero_grad()   # initialize gradient to be zero
        train_loss.backward()   # backpropagation
        optimizer.step()        # run one step

    print("epoch [{}/{}], training loss:{:.4f}".format(epoch+1, num_epochs, train_loss.data))

    for data in test_data:
        test_inputs, label = data
        test_inputs = test_inputs.view(test_inputs.size(0), -1)

        _, output_test = model(x=test_inputs)
        test_loss = criterion(output_test, test_inputs)

    print("epoch [{}/{}], validation loss:{:.4f}".format(epoch+1, num_epochs, test_loss.data))

    ''' 
    Compare with the lowest test_loss so far. 
    If the current test_loss is less than the lowest so far, continue.
    Otherwise stop current epoch 
    '''
    if test_loss <= validation_loss:
        validation_loss = test_loss
        # One can save the model!
        torch.save(model.state_dict(), './sim_autoencoder.pth')
        continue
    else:
        break #continue (for recording the best loss)


test_dataset = MNIST('./data', train = False, transform=img_transform, download=True)
test_data = DataLoader(test_dataset, batch_size=10000, shuffle=True)

#Test using MNIST
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

for data in test_data:
    '''
    Same as Validation? 
    Guess validation part should be replaced with validation data not test data.
    '''
    test_inputs, label = data
    test_inputs = test_inputs.view(test_inputs.size(0), -1)

    # forward
    latent, output = model(x=test_inputs)
    #print(latent)
    loss = criterion(output, test_inputs)
    print('test_loss(MSE): {:.4f}'.format(loss.data))

#print(latent.shape)
#print(label.shape)
#print(output.shape)
#print(output[10])
#print(output[2000].shape)

#print(test_inputs.shape)
#print(output.shape)

# TASK1: plot real data and output data (reconstructed)
temp = np.array(test_inputs[0])
temp = np.reshape(temp, (28,28))

test_np = np.array(test_inputs)
output_np = output.detach().numpy()

import scipy
from scipy.stats import norm

n=20    #20 numbers in a row, and a column
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

for i in range(n):
    for j in range(n):
        if j % 2 == 0:
            test_tmp = np.array(test_np[int(j/2) + 20*i])   #20*i !
            test_tmp = np.reshape(test_tmp, (28,28))        #real data
            figure[i*digit_size:(i+1)*digit_size, j*digit_size: (j+1)*digit_size] = test_tmp
        else:
            out_tmp = np.array(output_np[int(j/2)+20*i])
            out_tmp = np.reshape(out_tmp, (28,28))          #model's output data
            figure[i*digit_size:(i+1)*digit_size, j*digit_size:(j+1)*digit_size] = out_tmp

fig0 = plt.figure(figsize = (10,10))
plt.title('test_input(real)[left], model output[right] -> iterated')
plt.imshow(figure, cmap='gray')
plt.show()
fig0.savefig('AE MNIST')

#TASK2: visualizing latent space using matplotlib (dimension of the latent space should be 2)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
latent_np = latent.detach().numpy()
pca_output = pca.fit_transform(latent_np)

print(type(pca_output))

# Visualizing latent space when forward propagated test data

output_np = output.detach().numpy()
print(pca_output.shape)
print(output_np.shape)

#PCA result
fig1 = plt.figure(figsize=(10,10))
plt.scatter(pca_output[:,0], pca_output[:,1], c=label, cmap='nipy_spectral')
plt.clim(0,9)
plt.colorbar()
plt.title('A latent space of simple AE')
plt.show()
fig1.savefig('PCA.png')

#Auto Encoder result
num_epochs = 10

model_visual = autoencoder(latent_dim=2)
criterion_visual = nn.MSELoss()
optimizer_visual = torch.optim.Adam(model_visual.parameters())

#Training
validation_loss = 100
for epoch in range(num_epochs):
    for data in train_data:
        input_visual, label = data
        input_visual = input_visual.view(input_visual.size(0), -1)
        #forward
        latent_visual, output_visual = model_visual(x=input_visual)
        loss = criterion_visual(output_visual, input_visual)

        #backward (Weight update happens at this part)
        optimizer_visual.zero_grad()
        loss.backward()
        optimizer_visual.step()

    print("epoch [{}/{}], train loss:{:.4f}".format(epoch+1, num_epochs, loss.data))

    #validation
    for data in test_data: #data size = [10000 , 1 , 28 , 28]
        test_inputs, test_label = data
        test_inputs = test_inputs.view(test_inputs.size(0), -1)
        _, test_output = model_visual(x=test_inputs)
        test_loss = criterion_visual(test_output, test_inputs)

    print("epoch [{}/{}], validation loss:{:.4f}".format(epoch+1, num_epochs,test_loss.data))

    if test_loss <= validation_loss:
        validation_loss = test_loss
        torch.save(model_visual.state_dict(), './sim_autoencoder2.pth')
        continue
    else:
        break #continue

print(output_visual.shape)
print(test_inputs.shape)

#latent space
#print(model_visual.state_dict()) # check whether the network is trained or not

test_dataset = MNIST('./data', train=False, transform=img_transform, download=True)
test_data = DataLoader(test_dataset, batch_size=10000, shuffle=True)

for data in test_data:
    t_inputs, t_label = data
    t_inputs = t_inputs.view(t_inputs.size(0), -1)
    t_latent, t_output = model_visual(x=t_inputs)
    loss = criterion_visual(t_output, t_inputs)
    print('test_loss(MSE): {:.4f}'.format(loss.data))

print(t_inputs.shape)


latent_np = t_latent.detach().numpy()
output_np = t_output.detach().numpy()
print(latent_np.shape)
print(output_np.shape)
fig2 = plt.figure(figsize = (10,10))
plt.scatter(latent_np[:,0], latent_np[:,1], c=t_label, cmap= 'nipy_spectral')
plt.clim(0,9)
plt.colorbar()
plt.title('latent space of simple AE')
plt.show()
fig2.savefig('AE latent space')


# TASK: Uniform Sampling -> decode -> Compare with the above latent space plot

from torch.distributions import uniform
n = 20           # 20 digits for a row and a column
digit_size = 28
z_grid = torch.empty(n*n, 2)
z_grid.shape

x_len = latent[:,0].max() - latent[:,0].min()
x_section_len = float(x_len / n) # Sample 20 items from this interval length
y_len = latent[:,1].max() - latent[:,1].min()
y_section_len = float(y_len / n)

z_grid = torch.empty(n*n, 2)

for i in range(n):
    for j in range(n):
        # x value in i-th row, and y value in j-th row -? what does it mean?
        z_grid[i*n + j, 0] = torch.empty(1,).uniform_(float(latent[:,0].min()) + i*x_section_len, float(latent[:,0].min()) + (i+1)*x_section_len)
        z_grid[i*n + j, 1] = torch.empty(1,).uniform_(float(latent[:,1].max()) - j*y_section_len, float(latent[:,1].max()) - (j+1)*y_section_len)

# Pass the sampled data to Decoder
grid_out = model_visual.decoder(z_grid)
grid_out.shape
grid_out = grid_out.reshape(n, n, digit_size, digit_size)
grid_out.shape

figure = np.zeros((digit_size * n, digit_size * n))

for i in range(n):
    for j in range(n):
        temp = grid_out[i,j]
        temp = temp.detach().numpy()
        figure[j * digit_size:(j+1) * digit_size, i * digit_size:(i+1) * digit_size] = temp

plt.figure(figsize=(20,20))
plt.imshow(figure, cmap='gray')
plt.show()
