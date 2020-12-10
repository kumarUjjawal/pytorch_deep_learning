# Recurrent Neural Network Implementation in Pytorch with MNIST Dataset

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tochvision.datasets as dsets

# 1. Load Dataset

train_dataset = dsets.MNIST(root='./data',train=True, transforms=transforms.ToTensor(), download=True)

test_dataset = dsets.MNIST(root='./data', train=False, transforms=transforms.ToTensor(), download=True)

# 2. Make Dataset Iterable

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 3. Create Model Class

class RNNModel(nn.Module):
    def __init__(self, input_dim,hidden_dim, layer_dim, output_dim):
        super(RNNModel,self).__init__()
        
        # hidden dim
        self.hidden_dim = hidden_dim
        
        # number of hidden layers
        self.layer_dim = layer_dim

        # building rnn
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # readout layer
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        # initialize hidden dim with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        
        # detach hidden state to prevent exploding
        out,hn = self.rnn(x, h0.detach())

        # index hidden state of last time step

        out = self.fc(out[:,-1,:])

        return out


# 4. Instantiate Model Class

input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = RNNModel(input_dim,hidden_dim,layer_dim,output_dim)
model.to(device) # gpu support

# 5. Instantiate Loss Class

criterion = nn.CrossEntropyLoss()

# 6. Instantiate Optimizer Class

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 7. Train Model

seq_dim = 28

iter = 0

for epoch in range(num_epochs):
    for i, (images,label) in enumerate(train_loader):
        model.train()
        
        # load images
        images = images.view(-1,seq_dim, input_dim).requires_grad_().to(device)

        # clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)

        # calculate loss
        loss = criterion(outputs,labels)

        # getting gradient w.r.t. parameters
        loss.backward()

        # updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            model.eval()
            # calculate accuracy
            correct = 0
            total = 0

            # iterate through test datasets
            for images,labels in test_loader:
                images = images.view(-1,seq_dim,input_dim).to(device)

                # forward pass

                outputs = model(images)

                # get predictions
                _,predicted = torch.max(outputs.data,1)

                # total number of labels
                total += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # print loss
            print('Iteration:{}. Loss:{}. Accuracy:{}'.format(iter, loss.item(),accuracy))



