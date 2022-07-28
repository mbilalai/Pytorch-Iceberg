import torch

X = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype = torch.float32)
Y = torch.tensor([2,0, 4,0, 6.0, 8.0], dtype = torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Predictions before training: {forward(5):.3f}')

learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    y_pred = forward(X)

    l = loss(Y, y_pred)

    l.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad

    #zero gradients

    w.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Predictions after training: f(5) = {forward(5):.3f}')


