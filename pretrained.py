import torch
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image

alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True) #44.5mil parameters
print(resnet)

preprocess = transforms.Compose([transforms.resize(256),
    transforms.CenterCrop(224)
    transforms.ToTensor()
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])

img = Image.open('path to the input image')
img.show()

img_t = preprocess(img)

batch_t = torch.unsqueeze(img_t, 0) # create a batch of the image
#to be inferred upon

resnet.eval() # puts the model in evaluation mode

#inference
out = resnet(batch_t)

with open('path to txt file.txt') as f:
    labels = [line.strip() for line in f.readlines()]

-, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 # softmax converts to probabilities/percentages
print(labels[index[0]], percentage[index[0]].item())

print("The output will be: ('golden retriever', 96.29334259033203")


_, indices = torch.sort(out, descending=True) # sorts the scores in descending order
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]] #prints the scores
