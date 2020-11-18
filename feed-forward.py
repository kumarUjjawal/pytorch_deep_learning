# Feed forward neural network with Pytorch

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

device = torch.device("cuda:0" if torch.cuda.is_avilabel() else "cpu")

# 1. Loading MNIST Dataset
train_dataset = dsets.MNIST(root='./data', train=True,
                            transforms=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', train=False,
                           transforms=transforms.ToTensor())

# 2. Make Dataset Iterable
batch_size = 100
n_iter = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


# 3. Create Model Class
class FNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNModel, self).__init__()

        # linear function
        self.fc1 = nn.Linear(input_dim,hidden_dim)

        # non-linearity
        self.sigmoid = nn.Sigmoid()

        # linear function(readout)
        self.fc2 = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
         # linear function
         out = self.fc1(x)

         # non-linearity
         out = self.sigmoid(out)

         # linear fuction(readout)
         out = self.fc2(out)
         
         return out


# 4. Instantiate Model Class

input_dim = 28*28
hidden_dim = 100
output_dim = 10

model = FNNModel(input_dim, hidden_dim, output_dim)
model.to(device)
# 5. Instantiate Loss Class

criterion = nn.CrossEntropyLoss()

# 6. Instantiate Optimizer Class

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 7. Train Model
iter = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1,28*28).requires_grad_()

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            correct = 0
            total = 0
            for images,labels in test_loader:
                images = images.view(-1,28*28).requires_grad_()
                outputs = model(images)
                _,predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                corect += (predicted == labels).sum()
            accuracy = 100 * correct / total

            print('Iteration:{}. Loss:{}. Accuracy:{}'.format(iter, loss.item(),accuracy))








