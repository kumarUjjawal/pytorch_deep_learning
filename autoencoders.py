# Autoencoder with Pytroch using fashion MNIST dataset.

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy

figsize = (15,6)
plt.style('fivethirtyeight')

# 1. Load Dataset

train_dataset = dsets.FashionMNIST(root='./data',train=True,transforms=transforms.ToTensor(), download=True)

test_dataset = dsets.FashionMNIST(root='./data', train=False, transforms=transforms.ToTensor(), download=True)

# 2. Data Loader

batch_size = 100
n_iters = 5000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DatatLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Sample: Boot
sample_num = 0
show_img = train_dataset[sample_num][0].numpy().reshape(28,28)
label = train_dataset[sample_num][1]
print(f'Label {label}')
plt.imshow(show_img, cmap='gray')

# 3. Create Model Class

class Autoencoders(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # encoder: affine function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # decoder: affine function
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        # encoder: affine function
        out = self.fc1(x)
        # encoder: non-linear function
        out = F.leaky_relu(out)

        # decoder: affine function
        out = self.fc2(out)
        # decoder: non-linear function
        out = torch.sigmoid(out)

        return out

# 4. Instantiate Model Class

input_dim = 28*28
hidden_dim = int(input_dim * 1.5)
output_dim = input_dim

model = Autoencoder(input_dim, hidden_dim, output_dim)

# 5. Instantiate Loss Class

criterion = nn.MSELoss()

# 6. Instantiate Optimizer Class

learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# 7. Train Model

idx = 0

dropout = nn.Dropout(0.5)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.view(-1,28*28).requires_grad_()

        noisy_images = dropout(torch.ones(images.shape))*images

        # clear gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(noisy_images)

        # calculate loss
        loss = criterion(outputs,images)

        # getting gradients
        loss.backward()

        optimizer.step()

        idx += 1

        if idx % 500 == 0:
            total_test_loss = 0
            total_samples = 0

            for images,labels in test_loader:
                noisy_images = dropout(torch.ones(images.shape))*images

                outputs = model(noisy_images.view(-1,28*28))

                test_loss = criterion(outputs,images.view(-1,28*28))

                total_samples += labels.size(0)

                total_test_loss += test_loss

            mean_test_loss = total_test_loss / total_samples

            # print loss

            print(f'Iteration: {idx}. Average Test Loss: {mean_test_loss.item()}.')





