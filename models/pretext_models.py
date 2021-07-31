# -*- coding: utf-8 -*-
'''
Implement of MARTR GANs

Version 1.0  2020-09-09 23:56:05
by QiJi Refence: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .BasicModule import BasicModule


# class Context_Encoder(BasicModule):
class Context_Encoder0(nn.Module):
    def __init__(self, net, in_size=224):
        super().__init__()
        self.model_name = 'Context_encoder0_' + net.model_name
        # self.insize = in_size

        # Encoder
        self.feature = net.feature  # .copy()

        # replace Relu to LeakyRelu
        for name, module in self.feature.named_children():
            if isinstance(module, nn.ReLU):
                module = nn.LeakyReLU(0.2)

        self.out_dim = net.out_dim

        # Channel-wise fully-connected layer
        # if 'resnet' in net.model_name:
        #     md_size = 7
        # elif 'alexnet' in net.model_name:
        #     md_size = 6
        # self.channel_wise_fc = nn.Parameter(
        #     torch.rand(self.out_dim, md_size*md_size, md_size*md_size)
        # )
        # nn.init.normal_(self.channel_wise_fc, 0., 0.005)
        # self.dropout_cwfc = nn.Dropout(0.5, inplace=True)
        # self.conv_cwfc = nn.Conv2d(self.out_dim, self.out_dim, 1)

        # Decoder
        # if self.out_dim == 2048:
        #     next_dim = 512
        # else:
        #     next_dim = 256

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(self.out_dim, 128, 5, 2,
                               padding=2, output_padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # size: 128 x 11 x 11
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2,
                               padding=2, output_padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # size: 64 x 21 x 21
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 5, 2,
                               padding=2, output_padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # size: 64 x 41 x 41
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, 2,
                               padding=2, output_padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # size: 32 x 81 x 81
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 5, 2,
                               padding=2, output_padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            # size: 3 x 161 x 161
            # reszie to 227, 227
            nn.Tanh()
        )

    def forward(self, x):
        insize = x.size()[2:4]
        x = self.feature(x)  # s/32

        # N, C, H, W = x.size()[:]
        # x = x.view(N, C, -1)  # [N,C,H,W] -> [N,C,HW]
        # x = x.permute(1, 0, 2)  # [N,C,HW] -> [C,N,HW]
        # x = torch.bmm(x, self.channel_wise_fc)
        # x = x.permute(1, 0, 2)
        # x = self.dropout_cwfc(x)
        # x = x.view(N, C, H, W)
        # x = self.conv_cwfc(x)

        x = self.decoder1(x)  # s/16
        x = self.decoder2(x)  # s/8
        x = self.decoder3(x)  # s/4
        x = self.decoder4(x)  # s/2
        x = self.decoder5(x)  # s/1
        # x = nn.functional.interpolate(x, size=self.insize, mode='nearest')
        x = nn.functional.interpolate(x, size=insize, mode='nearest')
        return x


class Context_Encoder(nn.Module):
    def __init__(self, net, in_channels=3):
        super().__init__()
        self.model_name = 'Context_encoder2_' + net.model_name
        # self.insize = in_size
        self.out_dim = net.out_dim
        # Encoder
        self.conv1 = nn.Sequential(
            net.feature.conv1,
            net.feature.bn1,
            net.feature.relu,
        )  # 64, s/2
        self.layer1 = net.feature.layer1  # 256, s/4
        self.layer2 = net.feature.layer2  # 512, s/8
        self.layer3 = net.feature.layer3  # 1024, s/8
        self.layer4 = net.feature.layer4  # 2048, s/16

        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(2048, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 5, 2,
                               padding=2, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256+1024, 256, 5, 2,
                               padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256+512, 256, 3, 2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, 2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=7, padding=3, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        s_1 = x.size()[2:4]
        x = self.conv1(x)  # s/2
        # s_2 = x.size()[2:4]
        # x = self.maxpool(x)
        e1 = self.layer1(x)  # 256, s/2
        e2 = self.layer2(e1)  # 512, s/4
        s_4 = x.size()[2:4]
        e3 = self.layer3(e2)  # 1048, s/8
        s_8 = e3.size()[2:4]
        e4 = self.layer4(e3)  # 2048, s/16
        e4 = self.bottleneck_conv(e4)  # 256

        # Decode
        d4 = self.decoder4(e4)  # 256, s/8
        if d4.shape[2:4] != e3.shape[2:4]:
            d4 = F.interpolate(d4, size=s_8, mode='nearest')
        d4_cat = torch.cat((d4, e3), dim=1)  # 256+1024

        d3 = self.decoder3(d4_cat)  # 256, s/4
        if d3.shape[2:4] != e2.shape[2:4]:
            d3 = F.interpolate(d3, size=s_4, mode='nearest')
        d3_cat = torch.cat((d3, e2), dim=1)  # 256+512

        d2 = self.decoder2(d3_cat)  # 256, s/2
        d1 = self.decoder1(d2)  # 64, s/1
        if d1.shape[2:4] != s_1:
            d1 = F.interpolate(d1, size=s_1, mode='nearest')
        x = self.conv_final(d1)
        # x = nn.functional.interpolate(x, size=self.insize, mode='nearest')
        return x


class ColorizationNet(nn.Module):
    def __init__(self, net, in_channels=3):
        super().__init__()
        self.model_name = 'Colorization_' + net.model_name
        # self.insize = in_size
        self.out_dim = net.out_dim
        # Encoder
        self.conv1 = nn.Sequential(
            net.feature.conv1,
            net.feature.bn1,
            net.feature.relu,
            # net.feature.maxpool
        )  # 64, s/2
        self.layer1 = net.feature.layer1  # 256, s/4
        self.layer2 = net.feature.layer2  # 512, s/8
        self.layer3 = net.feature.layer3  # 1024, s/8
        self.layer4 = net.feature.layer4  # 2048, s/16

        self.up0 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(32, in_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        s_1 = x.size()[2:4]
        x = self.conv1(x)  # s/2
        s_2 = x.size()[2:4]
        # x = self.maxpool(x)
        x = self.layer1(x)  # 256, s/2
        x = self.layer2(x)  # 512, s/4
        s_4 = x.size()[2:4]
        x = self.layer3(x)  # 1048, s/8
        s_8 = x.size()[2:4]
        x = self.layer4(x)  # 2048, s/16

        # Decode
        x = F.interpolate(x, size=s_8, mode='nearest')
        x = self.up0(x)  # 512, s/8
        x = self.up1(x)  # 256, s/8
        x = F.interpolate(x, size=s_4, mode='nearest')
        x = self.up2(x)  # 128, s/4
        x = F.interpolate(x, size=s_2, mode='nearest')
        x = self.up3(x)  # 64, s/2
        x = F.interpolate(x, size=s_1, mode='nearest')
        x = self.up4(x)  # 32, s/1
        x = self.up5(x)  # 32, s/1

        return x


def weights_init_for_gan(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


class Generator_GAN2(nn.Module):
    ''' Generator of MARTA GANs, which modeified from DCGANs. '''
    def __init__(self, out_channels):
        super().__init__()
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )  # 512, 4x4, project and reshape
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )  # 256, 8x8
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )  # 128, 16x16
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )  # 64, 32x32
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )  # 32, 64x64
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )  # 16, 128x128
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )  # out_channels, 256x256

        for m in self._modules:
            weights_init_for_gan(self._modules[m])

    # forward method
    def forward(self, x):
        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        return x


class Discriminator_GANs2(torch.nn.Module):
    def __init__(self, num_classes=1, in_channels=3, supervised=False):
        super().__init__()
        self.supervised = supervised

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(True)
        )  # s/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True)
        )  # s/4
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )  # s/8
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True)
        )  # 128, s/16
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True)
        )  # 256, s/32
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True)
        )  # 512, s/64

        self.classifier = nn.Linear((128+256+512)*4*4, num_classes)

        for m in self._modules:
            weights_init_for_gan(self._modules[m])

    # forward method
    def forward(self, x):
        N = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        e4 = self.conv4(x)  # 128, s/16
        e5 = self.conv5(e4)  # 256, s/32
        e6 = self.conv6(e5)  # 512, s/64

        e4 = F.max_pool2d(e4, 4)
        e5 = F.max_pool2d(e5, 2)
        x = torch.cat([e4, e5, e6], dim=1)
        x = x.view(N, -1)
        logist = self.classifier(x)

        if self.supervised:  # mode for unsupervised learning
            return x, torch.sigmoid(logist)
        else:
            return logist


class Generator_GAN(nn.Module):
    ''' Generator of MARTA GANs, which modeified from DCGANs. '''
    def __init__(self, out_channels):
        super().__init__()
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )  # 512, 4x4, project and reshape
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )  # 256, 8x8
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )  # 128, 16x16
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )  # 64, 32x32
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )  # 32, 64x64
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )  # 16, 128x128
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )  # out_channels, 256x256

        for m in self._modules:
            self.weights_init(self._modules[m])

    def weights_init(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            # m.weight.data.normal_(0.0, 0.02)
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # forward method
    def forward(self, x):
        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        return x


class Discriminator_GANs(torch.nn.Module):
    def __init__(self, num_classes=1, in_channels=3, supervised=False):
        super().__init__()
        self.supervised = supervised

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2)
        )  # s/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )  # s/4
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )  # s/8
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )  # 128, s/16
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )  # 256, s/32
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )  # 512, s/64

        self.classifier = nn.Linear((128+256+512)*4*4, num_classes)

        for m in self._modules:
            self.weights_init(self._modules[m])

    def weights_init(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            # m.weight.data.normal_(0.0, 0.02)
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # forward method
    def forward(self, x):
        N = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        e4 = self.conv4(x)  # 128, s/16
        e5 = self.conv5(e4)  # 256, s/32
        e6 = self.conv6(e5)  # 512, s/64

        e4 = F.max_pool2d(e4, 4)
        e5 = F.max_pool2d(e5, 2)
        x = torch.cat([e4, e5, e6], dim=1)
        x = x.view(N, -1)
        logist = self.classifier(x)

        if self.supervised:  # mode for unsupervised learning
            return x, torch.sigmoid(logist)
        else:
            return logist


class LRN(nn.Module):
    def __init__(self,
                 local_size=1,
                 alpha=1.0,
                 beta=0.75,
                 across_channels=True):
        super(LRN, self).__init__()
        self.across_channels = across_channels
        if across_channels:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int(
                                            (local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.across_channels:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class Jigsawer(nn.Module):
    def __init__(self, net, num_classes, puzzle):
        super().__init__()
        self.out_dim = net.out_dim

        self.feature = nn.Sequential(
            net.feature.conv1,  # s/2
            net.feature.bn1,
            net.feature.relu,

            net.feature.maxpool,  # s/4
            # LRN(5, alpha=0.0001, beta=0.75),
            net.feature.layer1,
            # LRN(5, alpha=0.0001, beta=0.75),
            net.feature.layer2,  # s/8
            net.feature.layer3,  # s/16
            net.feature.layer4,  # s/32

            # bottleneck conv layer
            nn.Conv2d(self.out_dim, 512, 3, 1, 1),  # compress dim
            nn.AdaptiveAvgPool2d((2, 2))
        )  # 2 x 2
        # self.maxpool = net.feature.maxpool
        # self.lrn1 = LRN(5, alpha=0.0001, beta=0.75)
        # self.layer1 = net.feature.layer1
        # self.lrn2 = LRN(5, alpha=0.0001, beta=0.75)
        # self.layer2 = net.feature.layer2
        # self.layer3 = net.feature.layer3
        # self.layer4 = net.feature.layer4
        # self.bottleneck_conv = nn.Conv2d(self.out_dim, 512, 1)

        self.fc5 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.fc6 = nn.Sequential(
            nn.Linear(512 * (puzzle**2), 2048),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        ''' Take  '''
        N, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(T):
            z = self.feature(x[i])  # 2x2
            z = self.fc5(z.view(N, -1))
            z = z.view([N, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.fc6(x.view(N, -1))
        logist = self.classifier(x)
        return logist


def test():
    from .scene_net import Scene_Base
    # net = Scene_Base(2, 3, 'resnet50')
    # input = torch.rand((2, 3, 224, 224))
    # input = torch.rand((2, 3, 128, 128))
    # netG = Context_Encoder(net, 3)

    # output = netG(input)

    # input = torch.rand((2, 9, 3, 64, 64))
    # jigsawer = Jigsawer(100, net)
    # output = jigsawer(input)

    # input = torch.rand((2, 4, 256, 256))
    z_ = torch.randn((2, 100, 1, 1))
    netD = Discriminator_GANs(4)
    netG = Generator_GAN(4)
    output = netG(z_)
    y = netD(output)


if __name__ == "__main__":
    pass
