# Implementing Autoencoder in Pytorch

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt

# convert vectors to image
def to_img(x):
    x = 0.5 * (x+1)
    x = x.view(x.size(0), 28,28)
    return n


# display routine
def display_images(in_, out, n=1):
    for n in range(n):
        if in_ is not None:
            in_pic = to_img(in_.cpu().data)
            plt.figure(figsize=(16,6))
            for i in range(4):
                plt.subplo(1,4,i+1)
                plt.imshow(in_pic[i+4*N])
                plt.axis('off')
            out_pic = to_img(out.cpu().data)
            plt.figure(figsize=(16,6))
            for i in range(4):
                plt.subplot(1,4,i+1)
                plt.imshow(out_pic[i+4*N])
                plt.axis('off')

# define data loading step

batch_size = 256

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

dataset = MNIST('./data'. transform=img_transform, download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model architecture and recosntruction loss

d = 30

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, d), nn.Tanh())

        self.decoder = nn.Sequential(nn.Linear(d, 28 * 28), nn.Tanh())
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder.to(device)
criterion = nn.MSELoss()

# configure the optimizer
learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# train standard autoencoder

num_epochs = 20
for epoch in range(num_epochs):
    for data in dataloader:
        img,_ = data
        img = img.to(device)
        img = img.view(img.size(0), -1)

        output = model(img)
        loss = criterion(output, img.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'epoch [{epoch + 1}/{num_epoch}], loss: {loss.item():.4f}')
    display_images(None, output)

# visualize a few kernels of the encoder
display_images(None, model.encoder[0].weight, 5)


# comapre the autoencoder inpainting capabilities vs OpenCV

from cv2 import inpaint, INPAINT_NS, INPAINT_TELEA

dst_telea = list()
dst_ns = list()

for i in range(3,7):
    corrupted_img = ((img_bad.data.cpu()[i].view(28,28) / 4 + 0.5) * 255).byte().numpy()
    mask = 2 - noise.cpu()[i].view(28,28).byte().numpy()
    dst_telea.append(inpaint(corrupted_img, mask, 3, INPAINT_NS))

tns_telea = [torch.from_numpy(d) for d in dst_telea]
tns_ns = [torch.from_numpy(d) for d in dst_ns]

telea = torch.stack(tns_telea).float()
ns = torch.stack(tns_ns).float()

# compare the results
with torch.no_grad():
    display_images(noise[3:7], img_bad[3:7])
    display_images(img[3:7], output[3:7])
    display_images(TELEA,NS)


