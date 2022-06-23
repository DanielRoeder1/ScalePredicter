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