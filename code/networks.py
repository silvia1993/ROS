import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
from torchvision import models
import os
import numpy as np
from utilities import *
import torch.nn.functional as F
import scipy.io as sio


class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
    def output_num(self):
        pass
resnet_dict = {"resnet50":models.resnet50}
    
class ResNetFc(BaseFeatureExtractor):
    def __init__(self,device, model_name='resnet50', normalize=True):
        super(ResNetFc, self).__init__()
        self.model_resnet = models.resnet50(pretrained=True)
        self.normalize = normalize
        self.mean = False
        self.std = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features
        self.device = device
        self.fc = nn.Linear(self.__in_features, self.__in_features)
        self.bn_sharedfc = nn.BatchNorm1d(self.__in_features)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).to(self.device)
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).to(self.device)
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)       
        return x

    def output_num(self):

        return self.__in_features,1024

        
class VGGFc(BaseFeatureExtractor):
    def __init__(self,device, model_name='vgg19',normalize=True):
        super(VGGFc, self).__init__()
        self.model_vgg = models.vgg19(pretrained=True)

        self.normalize = normalize
        self.mean = False
        self.std = False
        model_vgg = self.model_vgg        
        mod = list(model_vgg.features.children())
        self.features = nn.Sequential(*mod)
        mod2 = list(model_vgg.classifier.children())[:-1]
        self.classifier = nn.Sequential(*mod2)
        self.__in_features = 4096
        self.device = device

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).to(self.device)
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).to(self.device)
        return self.std

    def forward(self, x):
        
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)        
        x = self.classifier(x)
        
        return x

    def output_num(self):

        return self.__in_features
        
class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256,vgg=None):
        super(CLS, self).__init__()
        self.vgg = vgg
        if bottle_neck_dim:
            if not vgg:
                self.bottleneck = nn.Linear(in_dim[0], bottle_neck_dim)
                self.fc = nn.Linear(bottle_neck_dim, out_dim)
                self.main = nn.Sequential(
                    self.bottleneck,
                    nn.Sequential(
                        nn.BatchNorm1d(bottle_neck_dim),
                        nn.LeakyReLU(0.2, inplace=True),
                        self.fc
                    ),
                    nn.Softmax(dim=-1)
                    )                    
            else:
                self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
                self.fc = nn.Linear(bottle_neck_dim, out_dim)
                self.main = nn.Sequential(
                    self.bottleneck,
                    nn.Sequential(
                        nn.BatchNorm1d(bottle_neck_dim),
                        nn.LeakyReLU(0.2, inplace=True),
                        self.fc
                    ),
                    nn.Softmax(dim=-1)
                    )

    def forward(self, x):
                
        out_last = [x]
        x_last = x
        for module in self.main.children():
            x_last = module(x_last)
            out_last.append(x_last)

        return out_last

    
class Discriminator(nn.Module):
    def __init__(self, n=None,n_s = None,vgg=None):
        super(Discriminator, self).__init__()
        self.n = n
        def f():
            if vgg:
                return nn.Sequential(
                    nn.Linear(4096*2, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(256, n_s))                                              
            else:
                return nn.Sequential(
                    nn.Linear(2048*2, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(256, n_s))                   

        def f_feat():
            if vgg:
                return nn.Sequential(
                    nn.Linear(4096*2, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2, inplace=True))  
            else:
                return nn.Sequential(
                    nn.Linear(2048*2, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2, inplace=True))                   
               
        for i in range(n):
            self.__setattr__('discriminator_%04d'%i, f())
            self.__setattr__('discriminator_feat_%04d'%i, f_feat())
    
    def forward(self, x):

        outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        outs_feat = [self.__getattr__('discriminator_feat_%04d'%i)(x) for i in range(self.n)]

        return torch.cat(outs, dim=-1),torch.cat(outs_feat, dim=-1)