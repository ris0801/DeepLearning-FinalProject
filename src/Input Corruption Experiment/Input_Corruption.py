import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from torchsummary import summary
import pandas as pd

transform = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(28),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./', train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=32, shuffle=True, num_workers=2,pin_memory=True)
testloader = torch.utils.data.DataLoader(testset,batch_size=32, shuffle=False, num_workers=2,pin_memory=True)

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvModule, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
 
class InceptionModule(nn.Module):
    
    def __init__(self, in_channels, f_1x1, f_3x3):
        super(InceptionModule, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvModule(in_channels, f_1x1, kernel_size=1, stride=1, padding=0)
        )
        
        self.branch2 = nn.Sequential(
            ConvModule(in_channels, f_3x3, kernel_size=3, stride=1, padding=1)
        )
                
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], 1)


class DownsampleModule(nn.Module):
    def __init__(self, in_channels, f_3x3):
        super(DownsampleModule, self).__init__()
    
        self.branch1 = nn.Sequential(ConvModule(in_channels, f_3x3, kernel_size=3, stride=2, padding=0))
        self.branch2 = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], 1)
   
class InceptionSmall(nn.Module):
    def __init__(self, num_classes = 10):
        super(InceptionSmall, self).__init__()
        
        self.conv1 = ConvModule(in_channels =3,out_channels=96, kernel_size=3, stride=1, padding=0)
        self.inception1 = InceptionModule(in_channels=96,f_1x1=32,f_3x3=32)
        self.inception2 = InceptionModule(in_channels=64,f_1x1=32,f_3x3=48)
        self.down1 = DownsampleModule(in_channels=80,f_3x3=80)
        self.inception3 = InceptionModule(in_channels=160,f_1x1=112,f_3x3=48)
        self.inception4 = InceptionModule(in_channels=160,f_1x1=96,f_3x3=64)
        self.inception5 = InceptionModule(in_channels=160,f_1x1=80,f_3x3=80)
        self.inception6 = InceptionModule(in_channels=160,f_1x1=48,f_3x3=96)   
        self.down2 = DownsampleModule(in_channels=144,f_3x3=96)
        self.inception7 = InceptionModule(in_channels=240,f_1x1=176,f_3x3=160)
        self.inception8 = InceptionModule(in_channels=336,f_1x1=176,f_3x3=160)
        self.meanpool = nn.AdaptiveAvgPool2d((7,7))
        self.fc = nn.Linear(16464, num_classes)
       
    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.down1(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.down2(x)
        x = self.inception7(x)
        x = self.inception8(x)
        x = self.meanpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

def get_device():
  if torch.cuda.is_available():
      return torch.device('cuda')
  else:
      return torch.device('cpu')
device = get_device()
device

def get_gaussian_image(batch, w, h):
    # CIFAR 10 mean and std in RGB format. Source: https://github.com/facebookarchive/fb.resnet.torch/issues/180
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    out = list()
    for i in range(batch):
        r_channel = torch.normal(mean[0], std[0], size=(w, h)).unsqueeze(0)
        g_channel = torch.normal(mean[1], std[1], size=(w, h)).unsqueeze(0)
        b_channel = torch.normal(mean[2], std[2], size=(w, h)).unsqueeze(0)
        out.append(torch.cat((r_channel, g_channel, b_channel), dim=0).unsqueeze(0))
    return torch.cat(out, dim=0)

def eval_model(model,trainloader,testloader):
  model.eval()
  correct = 0
  total = 0
  for i, data in enumerate(testloader, 0):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, dim=1)
    total += labels.size(0)
    correct += (preds == labels).sum().item()
  acc_v = (correct / total)

  correct = 0
  total = 0            
  for i, data in enumerate(trainloader, 0):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, dim=1)
    total += labels.size(0)
    correct += (preds == labels).sum().item()
  acc_t = (correct / total)
  return acc_t,acc_v

def fit(epoch, model, trainloader, testloader, optimizer,scheduler,name='model', random_shuffle=False, shuffled_pixels=False, random_pixels=False, gaussian_noise=False):
  history_t = []
  history_v = []
  history_loss = []
  step_count = 0
  criterion = nn.CrossEntropyLoss().to(device)
  if random_pixels:
    rng = np.random.default_rng()
  for epoch in range(epoch):  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)
      if random_shuffle:
        # The permutation should be the same for every epoch
        torch.manual_seed(0)
        labels = labels[torch.randperm(len(labels))]
      if shuffled_pixels:
        data_ = inputs.cpu().numpy()
        # Resetting the random generator every time to generate the same set of permutation
        rng = np.random.default_rng(26)
        data_perm = rng.permutation(data_, axis=2)
        data = torch.from_numpy(data_perm)
      if random_pixels:
        data_ = inputs.cpu().numpy()
        # Generating a different set of permutation every time
        data_perm = rng.permutation(data_, axis=2)
        data = torch.from_numpy(data_perm)
      if gaussian_noise:
        batch = inputs.shape[0]
        w = inputs.shape[-1]
        h = inputs.shape[-2]
        data = get_gaussian_image(batch, w, h)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      step_count += 1
      if step_count%1000==0:
        acc_t,acc_v = eval_model(model,trainloader,testloader)
        model.train()
        history_t.append(acc_t)
        history_v.append(acc_v)
        log_ = str(step_count)+","+str(acc_t)+","+str(acc_v)+ "," + str(running_loss/1000)+"\n"
        with open("{}.log".format(name), "a") as f:
          f.write(log_)
    print("Epoch {} | loss: {:.4f} | Train acc: {:.4f} | Val acc: {:.4f}".format(epoch+1, running_loss/1000,acc_t, acc_v))
    history_loss.append(running_loss/1000)
    scheduler.step()
  return model,history_t,history_v, history_loss
net = InceptionSmall().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
trained_model,history_t1,history_v1, history_loss = fit(100, net, trainloader, testloader, optimizer,scheduler, name='normal_labels')
plt.plot(list(history_loss), label='true labels')
plt.xlabel('Training Steps')
plt.ylabel('loss')
plt.title('Inception Small on Cifar-10')
plt.show()
net = InceptionSmall().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=2, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
trained_model,history_t1,history_v1, history_loss_random = fit(100, net, trainloader, testloader, optimizer,scheduler, name='random_labels', random_shuffle=True)

net = InceptionSmall().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
trained_model,history_t1,history_v1, history_loss_shuffled_pixels = fit(100, net, trainloader, testloader, optimizer,scheduler, name='shuffled_pixels', shuffled_pixels=True)

plt.plot(list(history_loss_shuffled_pixels), label='shuffled pixels')
plt.xlabel('Training Steps')
plt.ylabel('loss')
plt.title('Inception Small on Cifar-10')
plt.show()


net = InceptionSmall().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
trained_model,history_t1,history_v1, history_loss_random_pixels = fit(100, net, trainloader, testloader, optimizer,scheduler, name='random_pixels', random_pixels=True)

plt.plot(list(history_loss_random_pixels), label='random pixels')
plt.xlabel('Training Steps')
plt.ylabel('loss')
plt.title('Inception Small on Cifar-10')
plt.show()

net = InceptionSmall().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

trained_model,history_t1,history_v1, history_loss_gaussian_noise = fit(100, net, trainloader, testloader, optimizer,scheduler, name='gaussian_noise', gaussian_noise=True)
plt.plot(list(history_loss_gaussian_noise), label='Gaussian Noise')
plt.xlabel('Training Steps')
plt.ylabel('loss')
plt.title('Inception Small on Cifar-10')
plt.show()


