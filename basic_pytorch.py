import torch

x = torch.randn(3, requires_grad=True)
print(x)
y = x+2

z = y*y*2
#z = z.mean()
print(z)

v = torch.tensor([1.0, 2.0, 0.1])
z.backward(v)

print(x.grad)

#stop torch from computing gradients and tracking history

#x.requires_grad_()
#y = x.detach()
#with torch.no_grad():
    #y = x + 2
    #print(y)

#weights.grad.zero_()
#incase of builtin optim

optimizer = torch.optim.SGD(weights, lr=1e3)
optimizer.step()
optimizer.zero_grad()
