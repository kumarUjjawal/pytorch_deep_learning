# Long Short Term Memory on MNIST Dataset Using Pytorch

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# 1. Load Dataset

train_dataset = dsets.MNIST(root='./data', train=True, transforms=transforms.ToTensor(),download=True)

test_dataset = dsets.MNIST(root='./data',train=False, transforms=transforms.ToTensor(), shuffle=True, download=True)

# 2. Make Dataset Iterable

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 3. Create Model Class

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, ouput_dim):
        super(LSTM,self).__init__()

        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim

        # lstm
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        # initialize hidden state with zeros
        h0 = torch.zeros(self.layers_dim, x.size(0), self.hidden_dim).requires_grad_()

        # initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        out,(hn,cn) = self.lstm(x,(h0.detach(),c0.detach()))
        
        # index hidden state of last time step
        out = self.fc(out[:,-1,:])

        return out



# 4. Instantiate Model Class

input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10

# GPU Use
device = torch.device("cuda:0" if torch.cuda.is_avialable() else "cpu")
model = LSTM(input_dim, hidden_dim, layer_dim, output_dim)
model.to(device)
# 5. Instantiate Loss Class

criterion = nn.CrossEntropyLoss()

# Instantiate Optimizer Class

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 7. Train Model

seq_dim = 28 # number of steps to unroll

iter = 0
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
        labels = labels.to(device) 

        # clear gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)

        loss = criterion(outputs,labels) # calculate loss

        # getting gradients
        loss.backward()

        # updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # calculate accuracy
            correct = 0
            total = 0
            for images,labels in test_loader:
                images = images.view(-1, seq_dim, input_dim).to(device)
                labels = labels.to(device)

                outputs = model(images)

                # get predictions
                _,predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            print('Iteration: {}. Loss:{}. Accuracy: {}'.format(iter, loss.item(), accuracy))


