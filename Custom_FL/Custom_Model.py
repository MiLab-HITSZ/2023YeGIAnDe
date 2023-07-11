# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time       : 2023/7/11 5:04 下午
# @Author     : Zipeng Ye
# @Affliction : Harbin Institute of Technology, Shenzhen
# @File       : Custom_Model.py

from torch import nn
import torchvision


class CustomResnet50_Dropout(nn.Module):
    def __init__(self, classes):
        super(CustomResnet50_Dropout, self).__init__()
        self.model = getattr(torchvision.models, 'resnet50')(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, classes),
        )

    def forward(self, x):
        return self.model(x)


class CustomResnet50(nn.Module):
    def __init__(self, classes):
        super(CustomResnet50, self).__init__()
        self.model = getattr(torchvision.models, 'resnet50')(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, classes)

    def forward(self, x):
        return self.model(x)


class CustomResnet34(nn.Module):
    def __init__(self, classes):
        super(CustomResnet34, self).__init__()
        self.model = getattr(torchvision.models, 'resnet34')(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, classes)

    def forward(self, x):
        return self.model(x)


class CustomResnet18(nn.Module):
    def __init__(self, classes):
        super(CustomResnet18, self).__init__()
        self.model = getattr(torchvision.models, 'resnet18')(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, classes)

    def forward(self, x):
        return self.model(x)


class CustomResnet18_Dropout(nn.Module):
    def __init__(self, classes):
        super(CustomResnet18_Dropout, self).__init__()
        self.model = getattr(torchvision.models, 'resnet18')(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, classes),
        )

    def forward(self, x):
        return self.model(x)
