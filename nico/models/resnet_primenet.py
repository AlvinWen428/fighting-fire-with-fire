import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet224 import BasicBlock, ResNet_Feature


class PrimeResNet(nn.Module):
    def __init__(self, block, layers, condition_activation='relu', num_classes=1000):
        super(PrimeResNet, self).__init__()
        self.feature_extractor = ResNet_Feature(block, layers, num_classes)
        self.coarse_classifier = nn.Linear(512 * block.expansion, num_classes)
        self.refine_classifier = nn.Linear(512 * block.expansion + num_classes, num_classes)
        self.condition_activation = condition_activation

    def activation_on_condition(self, condition):
        if self.condition_activation == 'relu':
            return F.relu(condition)
        elif self.condition_activation == 'softmax':
            return F.softmax(condition, dim=1)
        elif self.condition_activation is None:
            return condition
        else:
            raise ValueError

    def forward(self, x, key_input, train_mode=False):
        coarse_feature = self.feature_extractor(key_input)
        prime_variable = self.coarse_classifier(coarse_feature)
        input_condition = self.activation_on_condition(prime_variable)

        refine_feature = self.feature_extractor(x)
        refine_output = self.refine_classifier(torch.cat([refine_feature, input_condition], dim=1))
        if train_mode:
            return refine_output, prime_variable
        else:
            return refine_output


def primenet_resnet18(pretrained=False, **kwargs):
    model = PrimeResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise ValueError
    return model
