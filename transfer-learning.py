# Transfer learning Using Pytorch library on ImageNet dataset.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# model
transfer_model = models.resnet50(pretrained=True)

# freeze the parameters
for name,param in transfer_model.named_parameters():
    if ("bn" not in name):
        param.requires_grad = False


# replace the classifier
transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500), nn.ReLU(), nn.Dropout(), nn.Linear(500,2))

# training
def train(model,optimizer,loss_fn,train_loader,val_loader,epochs=20,device='cpu'):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader():
            optimizer.zero_grad()
            inputs,targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.ste()
            training_loss  += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output),dim=1)[1],targets).view(-1)
            num_corrects += torch.sum(correct).item()
            num_examples += len(val_loader.dataset)

            print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch,training_loss,valid_loss, num_correct/num_examples))

# preprocess image

def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False
img_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229, 0.224, 0.225])
    ])

train_data_path = "./train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transform, is_valid_file=check_image)

val_data_path = "./val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=img_transform, is_valid_file=check_image)

batch_size = 64

train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# optimizer
transfer_model.to(device)
optimizer = optim.Adam(transfer_model.parameters(),lr=0.001)

train(transfer_model, optimizer, torch.nn.CrossEntropyLoss(),train_data_loader, val_data_loader, epochs=5, device=device)

# Find learning rate

def find_lr(model, loss_fn, optimizer, train_loader,init_value-1e-8, final_value=10.0, device='cpu'):
    number_in_epoch = len(train_loaser) - 1
    update_step = (final_value/init_value) ** (1/number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        inputs,targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # crash out if loss explodes
        if batch_num > 1 and loss 4 * best_loss:
            if (len(log_lrs) > 20):
                return log_lrs[10:-5], losses[10:-5]
            else:
                return log_lrs, losses
        # record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # store the values
        losses.append(loss.item())
        log_lrs.append((lr))

        # do the backward pass and optimize
        loss.backward()
        optimizer.step()

        # update the lr for the next step and store
        lr *= update_step
        optimizer.param_groupd[0]["lr"] = lr
    if(len(log_lrs) > 20):
        return log_lrs[10:-5], losses[10:-5]
    else:
        return log_lrs, losses

# learning rate
(lrs,losses) = find_lr(transfer_model, torch.nn.CrossEntropyLoss(),optimizer, train_data_loader, device=device)

# plot learning rate and loss
plt.plot(lrs,losses)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.show()

# Custom Transforms
# We'll create a lambda transform and a custom transform class.

def random_color_space(x):
    output = x.convert("HSV")
    return output

color_transform = transforms.Lambda(lambda x: random_color_space(x))

random_color_transform = torchvision.transforms.RandomApply([color_transform])

class Noise():
    def __init__(self,mean,stddev):
        self.mean = mean
        self.stddev = stddev
    def __call__(self, tensor):
        noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)
        return tensor.add_(noise)
    def __repr__(self):
        repr = f"{self.__class__.__name__}(mean={self.mean},stddev={self.stddev})"
        return repr

custom_tranform_pipeline = tranforms.Compose([random_color_transform, Noise(0.1, 0.05)])

# Ensembles
models_ensemble = [model.resnet50().to(device), models.resnet50().to(device)]

prediction = [F.softmax(m(torch.rand(1,3,224,224).to(device))) for m in models_ensemble]

avg_prediction = torch.stack(predictions).mean(0).argmax()

print(avg_prediction)

torch.stack(predictions)










