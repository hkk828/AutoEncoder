import torch
import torchvision

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root='./MNIST_data', train=True, transform=img_transform, download=False)
test_dataset= MNIST(root='./MNIST_data', train=False, transform=img_transform, download=False)

batch_size = 128

train_sampler = SubsetRandomSampler(list(range(55000)))
valid_sampler = SubsetRandomSampler(list(range(55000, 60000)))

train_data = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1, pin_memory=True)
valid_data = DataLoader(train_dataset, batch_size=5000, sampler=valid_sampler, num_workers=1, pin_memory=True)
test_data = DataLoader(test_dataset, batch_size=10000, num_workers=1, pin_memory=True, shuffle=True)

from torch import nn

input_size = 28 * 28
output_size = 10

# Refer FC
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax()
        )
    def forward(self, x):
        post_layer1 = self.layer1(x)
        post_layer2 = self.layer2(post_layer1)
        final = self.layer3(post_layer2)
        return final

class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()

        conv1 = nn.Conv2d(1,6,5,1)
        pool1 = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(6, 16, 5, 1)
        pool2 = nn.MaxPool2d(2)

        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(True),
            pool1,
            conv2, nn.BatchNorm2d(16),
            nn.ReLU(True),
            pool2
        )

        fc1 = nn.Linear(16*4*4, 120)
        fc3 = nn.Linear(120, 10)

        self.fc_module = nn.Sequential(
            fc1, nn.BatchNorm1d(120),
            nn.ReLU(True),
            #fc2, nn.ReLU(True),
            fc3
        )

        self.conv_module = self.conv_module.to(device)
        self.fc_module = self.fc_module.to(device)

    def forward(self, x):
        conv_result = self.conv_module(x)
        dim = 1

        for d in conv_result.size()[1:]:
            dim = dim * d

        conv_result = conv_result.view(-1, dim)
        output = self.fc_module(conv_result)

        return nn.functional.softmax(output, dim=1)

classifier = MyConvNet().to(device)
#classifier = MyNetwork().to(device)
cost_function = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(classifier.parameters())
#optimizer = torch.optim.SGD(classifier.parameters(), lr=0.005, momentum=0.9)

num_epoch = 25
validation_criterion = 10
best_network = None
for epoch in range(num_epoch):
    for data in train_data:
        input_train, label_train = data
        #input_train = input_train.view(input_train.size(0), -1)
        input_train = input_train.to(device)

        result_train = classifier(input_train)
        cost_train = cost_function(result_train, label_train.to(device))

        optimizer.zero_grad()
        cost_train.backward()
        optimizer.step()

    print("epoch: [{}/{}], train loss: {:.4f}".format(epoch+1, num_epoch, cost_train.data))

    for data in valid_data:
        input_valid, label_valid = data
        #input_valid = input_valid.view(input_valid.size(0), -1)
        input_valid = input_valid.to(device)

        result_valid = classifier(input_valid)
        cost_valid = cost_function(result_valid, label_valid.to(device))

    print("epoch: [{}/{}], validation loss: {:.4f}".format(epoch+1, num_epoch, cost_valid.data))

    if cost_valid <= validation_criterion:
        validation_criterion = cost_valid
        best_network = classifier


from sklearn.metrics import accuracy_score

for data in test_data:
    input_test, label_test = data
    #input_test = input_test.view(input_test.size(0), -1)
    input_test = input_test.to(device)
    label_test = label_test.to(device)

    result_test = classifier(input_test)
    result_test_best = best_network(input_test)
    cost_test = cost_function(result_test, label_test)
    cost_test_best = cost_function(result_test_best, label_test)

    print("test loss: {:.4f}".format(cost_test.data))
    print("test loss (best): {:.4f}".format(cost_test_best.data))

label_output = torch.max(result_test, 1)[1]
label_output_best = torch.max(result_test_best,1)[1]
accuracy = accuracy_score(label_output.detach().cpu().numpy(), label_test.detach().cpu().numpy())
accuracy_best = accuracy_score(label_output_best.detach().cpu().numpy(), label_test.detach().cpu().numpy())
print("test accuracy: {:.4f}".format(accuracy))
print("test accuracy (best): {:.4f}".format(accuracy_best))
