from turtle import forward
import torch.nn as nn
import torchvision
import math
import torch

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class ResNet(nn.Module):
    def __init__(self, path_pretrained= "", layers = 50):
        super(ResNet, self).__init__()

        if path_pretrained:
          checkpoint = torch.load(path_pretrained)
          pretrained_model = checkpoint['model']
          self.conv1 = pretrained_model._modules['conv1']
          self.bn1 = pretrained_model._modules['bn1']
          self.conv2 = pretrained_model._modules['conv2']
          self.bn2 = pretrained_model._modules['bn2']
          num_channels = 2048

        else:
          pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=True)
          self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
          self.bn1 = nn.BatchNorm2d(64)
          if layers <= 34:
              num_channels = 512
          elif layers >= 50:
              num_channels = 2048
          self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
          self.bn2 = nn.BatchNorm2d(num_channels//2)


        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # weight init
        weights_init(self.conv1)
        weights_init(self.bn1)
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)


        # Additional layers for classificaion
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.lin1 = nn.Linear(in_features =num_channels//2,out_features =  256)
        self.lin2 = nn.Linear(in_features = 256, out_features = 64)
        self.lin3 = nn.Linear(in_features = 64, out_features = 1)


    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)

        return x


class CnnRegressor(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(4, 20, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(20,20, kernel_size = 3, stride = 1, padding = 1)
        
        self.conv3 = nn.Conv2d(20, 40, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(40,40, kernel_size = 3, stride = 1, padding = 1)

        self.conv5 = nn.Conv2d(60, 60, kernel_size = 3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(60,60, kernel_size = 3, stride = 1, padding = 1)

        self.conv7 = nn.Conv2d(60, 80, kernel_size = 3, stride = 1, padding = 1)
        self.conv8 = nn.Conv2d(80,80, kernel_size = 3, stride = 1, padding = 1)

        self.conv7 = nn.Conv2d(80, 100, kernel_size = 3, stride = 1, padding = 1)
        self.conv8 = nn.Conv2d(100,100, kernel_size = 3, stride = 1, padding = 1)


        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        self.conv5.apply(weights_init)
        self.conv6.apply(weights_init)
        self.conv7.apply(weights_init)
        self.conv8.apply(weights_init)

        self.max = nn.MaxPool2d(2)
        self.activation = nn.ReLu()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.conv3 = nn.Conv2d()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.max(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.max(x)

        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = self.max(x)

        x = self.conv7(x)
        x = self.activation(x)
        x = self.conv8(x)
        x = self.activation(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
    

