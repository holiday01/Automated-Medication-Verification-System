import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, VGG11_Weights
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import numpy as np
start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(224,antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

batch_size = 16
dataset = torchvision.datasets.ImageFolder("./dataset/", transform=transform)
l = int(len(dataset)*0.8)
ll = len(dataset) - l
train_set, val_set = torch.utils.data.random_split(dataset, [l, ll])
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        self.l1 = nn.Linear(1000, 11)
    def forward(self, x):
        x = self.model(x)
        x = self.l1(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)

loss_save = []
for epoch in range(1000):  # loop over the dataset multiple times
    print(epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print(f'Epoch: {epoch + 1}, Loss: {running_loss:.5f}')
    loss_save.append(running_loss)
    if epoch > 20:
        torch.save(net.state_dict(), f'./model/model_weights_{epoch}.pt')
        if (loss_save[epoch-20] < np.array(loss_save[epoch-19:epoch+1])).all():
            print("early stop")
            break
    end = time.time()
    print(f'Finished Training {end-start}')


print(f'Finished Training {end-start}')
