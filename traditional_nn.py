import torch
from torch import nn

image = torch.randn(3, 10, 20)

print(len(image))

d0 = image.nelement()
print(d0)

class tradnn(nn.Module):
    def __init__(self,d0,d1,d2,d3):
        super().__init__()
        self.m0 = nn.Linear(d0, d1)
        self.m1 = nn.Linear(d1, d2)
        self.m2 = nn.Linear(d2, d3)

    def forward(self, x):
        a0 = x.view(-1) #Flatten the input tensor # z0 = x.view(-1)
        z1 = self.m0(x) # s1 = self.m0(x) ,  z1 = w1x0+b1
        a1 = torch.relu(z1) # z1 = torch.relu(s1) , a1 = relu(z1)
        z2 = self.m1(a1)  #s2 = self.m1(z1) , z2 = w2a1+b2
        a2 = torch.relu(z2) #z2 = torch.relu(s2) , a2 = relu(z2)
        z3 == self.m2(a2)#s3 = self.m2(z2) , z3 = w3a2+b3
        return z3

model = tradnn(d0,60,40,10)
print(model)
out = model(image)
print(out)
