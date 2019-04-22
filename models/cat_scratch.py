import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels,
                                 eps=0.001,
                                 momentum=0.1,
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResLayer, self).__init__()
        self.layer1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.layer2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.layer3 = BasicConv2d(64, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out += x
        out = self.relu(out)
        return out

class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()

        self.branch4x4_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.branch4x4_2 = BasicConv2d(256, 512, kernel_size=4, stride=2)

        self.branch6x6_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.branch6x6_2 = BasicConv2d(256, 256, kernel_size=(6, 6), stride=2, padding=1)
        self.branch6x6_3a = BasicConv2d(256, 512, kernel_size=1)
        self.branch6x6_3b = BasicConv2d(256, 512, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, 512, kernel_size=1)

    def forward(self, x):
        branch4x4 = self.branch4x4_1(x)
        branch4x4 = self.branch4x4_2(branch4x4)

        branch6x6 = self.branch6x6_1(x)
        branch6x6 = self.branch6x6_2(branch6x6)
        branch6x6a = self.branch6x6_3a(branch6x6)
        branch6x6b = self.branch6x6_3b(branch6x6)

        branch_pool = F.max_pool2d(x, kernel_size=4, stride=2)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch4x4, branch6x6a, branch6x6b, branch_pool]

        return torch.cat(outputs, 1)



class InceptionR(nn.Module):
    def __init__(self, in_channels):
        super(InceptionR, self).__init__()

        self.branch1x1_1 = BasicConv2d(in_channels, 32, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 32, kernel_size=5, padding=2)

        self.branch7x7_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7_4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.branch_pool = BasicConv2d(in_channels, 64, kernel_size=1) # 1x1 filter
    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7 = self.branch7x7_4(branch7x7)

        branch_pool_avg = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool_avg = self.branch_pool(branch_pool_avg)

        branch_pool_max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool_max = self.branch_pool(branch_pool_max)


        outputs = [branch1x1, branch3x3, branch5x5, branch7x7, branch_pool_avg, branch_pool_max]
        return torch.cat(outputs, 1)



class BreedClassifier(nn.Module):
    def __init__(self, im_size, n_classes):
        super(BreedClassifier, self).__init__()

        c = im_size[0] # channel
        h = im_size[1] # height
        w = im_size[2] # weight

        # Build a Model
        self.conv_1 = BasicConv2d(c, 32, 3, stride=2, padding=1)
        w = (w - 3 + 2 * 1) // 2 + 1
        h = (h - 3 + 2 * 1) // 2 + 1


        self.conv_2 = BasicConv2d(32, 32, 3, stride=1, padding=1)
        w = (w - 3 + 2 * 1) // 1 + 1
        h = (h - 3 + 2 * 1) // 1 + 1

        self.conv_3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        w = (w - 3 + 2 * 1) // 1 + 1
        h = (h - 3 + 2 * 1) // 1 + 1

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        w = (w - 2) // 2 + 1
        h = (h - 2) // 2 + 1

        self.conv_4 = BasicConv2d(64, 64, 3, stride=1, padding=1)
        w = (w - 3 + 2 * 1) // 1 + 1
        h = (h - 3 + 2 * 1) // 1 + 1

        self.conv_5 = BasicConv2d(64, 128, 3, stride=1, padding=1)
        w = (w - 3 + 2 * 1) // 1 + 1
        h = (h - 3 + 2 * 1) // 1 + 1

        self.conv_6 = BasicConv2d(128, 128, 3, stride=1, padding=1)
        w = (w - 3 + 2 * 1) // 1 + 1
        h = (h - 3 + 2 * 1) // 1 + 1

        self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        w = (w - 3) // 2 + 1
        h = (h - 3) // 2 + 1
        #################### Pre-Inception ###########################
        self.inception_0 = InceptionR(128)
        self.inception_1 = InceptionR(320)
        self.inception_2 = InceptionR(640)
        self.pool_3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        w = (w - 2) // 2 + 1
        h = (h - 2) // 2 + 1

        self.res_1 = ResLayer(320)
        self.res_2 = ResLayer(640)

        self.inception_3 = InceptionD(960)
        w = (w - 4) // 2 + 1
        h = (h - 4) // 2 + 1
        self.pool_4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        w = (w - 2) // 2 + 1
        h = (h - 2) // 2 + 1

        # Post Inception
        self.fc_1 = nn.Linear(2048 * w * h, n_classes)

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None

        x = self.conv_1(images)
        x = self.conv_2(x)
        x = self.conv_3(x)

        x = self.pool_1(x)

        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)

        x = self.pool_2(x)


        ########### Pre-Inception ########################
        x = self.inception_0(x)
        x_i = self.inception_1(x)
        x_r = self.res_1(x)
        x = torch.cat([x_i, x_r], 1)
        x_i = self.inception_2(x)
        x_r = self.res_2(x)
        x = torch.cat([x_i, x_r], 1)
        x = self.pool_3(x)
        x = self.inception_3(x)

        # Final Block
        x = self.pool_4(x)
        x = F.dropout(x, p=0.7, training=self.training)
        x = x.view(images.size(0), -1)
        scores = self.fc_1(x)

        return scores
