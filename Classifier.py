%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = 'assets/Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.compose([transforms.Resize(224),
                                       transforms.CenterCrop(223),
                                       transforms.ToTensor()])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=True)

for param in model.parameters():
    param.requires_grad() = False

model.Classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.ADAM(model.classifier.parameters(), lr=0.03)

model.to(device)

epochs = 1
steps = 0
print_stats = 5
train_loss = 0

for e in range(epochs):
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if steps %% print_stats == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                
                log_ps = model(images)
                val_loss = criterion(log_ps, labels)
                
                test_loss += val_loss.item
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equal = top_class == labels.view(*top_class.shape)
                
                accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
            
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Train Loss: {train_loss/print_stats:.3f}.. "
                  f"Test Loss: {test_loss/len(testloader):.3f}.. "
                  f"Accuracy: {accuracy/len(testloader):.3f}")
            train_loss = 0
            model.train()