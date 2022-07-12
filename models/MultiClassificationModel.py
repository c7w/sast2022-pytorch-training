import torch.nn as nn
from typing import List


class MultiClassificationModel(nn.Module):
    def __init__(self, num_categories=3):
        """
        We use a typical VGG16 network architecture, which could serve as a backbone for extracting global features
        from the input image.
        Then we declare `three` classification head, each is a binary classifier that output True or False
        with the input of features extracted by VGG19.
        """
        self.backbone = VGG16()
        self.cls_head = nn.ModuleList()
        for i in range(num_categories):
            self.cls_head.append(ClassificationHead())


    def forward(self, train_data):
        features = self.backbone(train_data)
        return torch.stack([head(features) for head in self.cls_head])


class VGG16(nn.Module):
    """
    Typical VGG16 Deep convolutional model.
    Copied from https://github.com/aaron-xichen/pytorch-playground/blob/master/imagenet/vgg.py.
    """
    def __init__(self):
        self.model = self._make_layers([64, 64, 'M',
                                        128, 128, 'M',
                                        256, 256, 256, 'M',
                                        512, 512, 512, 'M',
                                        512, 512, 512, 'M'])

    @classmethod
    def _make_layers(cls, cfg: List, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self):
        pass


class ClassificationHead(nn.Module):
    def __init__(self):
       self.linear = [
           nn.Linear(1024 / 32 * 768 / 32, 256),
           nn.Linear(256, 64),
           nn.Linear(64, 2),
       ]
        self.relu = nn.ReLU()
        self.softmax = nn.SoftMax()
        self.dropout = nn.Dropout()

    def forward(self, features):
        f1 = self.dropout(self.relu(self.linear[0](features)))
        f2 = self.dropout(self.relu(self.linear[1](features)))
        return self.softmax(self.linear[2](features))
