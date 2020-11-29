# Pytorch: Autograd

# Using Pytorch's automatica differentiation to automate the computation of backward passes in neural networks.


import torch

device = torch.device('cuda' if torch.cuda.is_availble() else 'cpu')

batch_size = 64
input_dim = 1000
hidden_dim = 100
output_dim = 10

# create random tensors to hold input and outputs
x = torch.randn(batch_size, input_dim, device=device)
y = torch.rand(batch_size, output_dim, device=device)

# create random tensors for weights
# requires_grad=True, if we want to compute gradients for these tensors during backward pass

w1 = torch.randn(input_dim, hidden_dim, device=device, requires_grad=True)
w2 = torch.randn(hidden_dim, output_dim, device=device, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # use autograd to compute the backward pass
    loss.backward()

    # update the weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # manually zero the gradients after running the backward pass
        w1.grad.zero_()
        w2.grad.zero_()
