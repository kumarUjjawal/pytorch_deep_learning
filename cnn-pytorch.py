# Convolutional Neural Network with Pytorch

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# 1. Load Dataset

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = dsets.MNIST(root='./data',train=False, transform=transforms.ToTensor(), download=True)

# 2. Make Dataset Iterable

batch_size = 100
n_iter = 3000
num_epochs = n_iter / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 3. Create Model Class

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()

        # conv1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # conv2
        self.cnn2 = nn.Conv2d(in_channel=16, out_channel=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # fully connected
        self.fc1 = nn.Linear(32*7*7, 10)

    def forward(self,x):
        # conv1
        out = self.cnn1(x)
        out = self.relu1(out)

        # maxpool 1
        out = self.maxpool(out)

        # conv2
        out = self.cnn2(out)
        out = self.relu2(out)

        # max pool 2
        out = self.maxpool2(out)

        # resize
        out = out.view(out.size(0),-1)

        # linear function
        out = self.fc1(out)

        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 4. Instantiate Model Class

model = CNNModel()
model.to(device)

# 5. Instantiate Loss Class

criterion = nn.CrossEntropyLoss()

# 6. Instantiate Optimizer Class

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# 7. Train Model

iter = 0
for epoch in range(num_epochs):
    for i, (image,labels) in enumerate(train_loader):
        # load images
        images = images.requres_grad_().to(device)
        labels = labels.to(device) 

        # cler gradients w.r.t. parameters
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)

        # calculate loss
        loss = criterion(outputs, labels)

        # get gradient w.r.t. parametsers
        loss.backward()

        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # calculate accuracy
            correct = 0
            total = 0
            # iterate through test dataset
            for image,labels in test_loader:
                images = images.requires_grad_().to(device)
                labels = labels.to(device)

                # forward pass
                outputs = model(images)

                # get predictions
                _,predicted = torch.max(outputs.data,1)

                # total number of labels
                total += labels.size(0)
                
                # total correct predictions
                if torch.cuda.is_available(): # use gpu
                    corect += (predicted == labels).sum()
                else:
                    correct += (precicted == labels).sum()
            accuracy = 100 * correct / total

            # print loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

             

