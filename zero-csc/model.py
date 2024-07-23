import torch
import torch.nn as nn
import numpy as np
import cv2
from loss import LossFunction
import torch.nn.functional as F


def make_kernel(f):
    kernel = np.zeros((2 * f + 1, 2 * f + 1), np.float32)
    for d in range(1, f + 1):
        kernel[f - d:f + d + 1, f - d:f + d + 1] += (1.0 / ((2 * d + 1) ** 2))

    return kernel / kernel.sum()



class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(    # 输入层
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):

        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)

        delta = input - fea

        return delta

class FeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(FeatureBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(3, out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

        self.layers = layers
        self.saturation_factor = 0.5

    def color_correction(self, x):
            img_np = x.detach().cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_corrected = np.zeros_like(img_np, dtype=np.uint8)
            for channel in range(img_np.shape[1]):
                channel_eq_hsv = cv2.cvtColor(cv2.cvtColor(img_corrected[0, channel], cv2.COLOR_GRAY2RGB),
                                              cv2.COLOR_RGB2HSV)
                channel_eq_hsv = channel_eq_hsv.astype(np.float32)
                channel_eq_hsv[:, :, 1] *= self.saturation_factor
                channel_eq_hsv[channel_eq_hsv[:, :, 1] > 255] = 255
                channel_eq_rgb = cv2.cvtColor(channel_eq_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                img_corrected[0, channel] = channel_eq_rgb[:, :, channel]

            img_corrected = torch.from_numpy(img_corrected).float() / 255.0

            return img_corrected

    def forward(self, x):
        # 在这里应用色彩校正操作
        out = self.color_correction(x)
        out = out.float()
        out = out.squeeze(0) if out.dim() == 5 else out.squeeze()
        out = out.cuda()

        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        for conv in range(self.layers):
            out1 = self.conv1(out)
            out2 = self.relu(out1)
            out3 = self.conv2(out2)
            out4 = self.relu(out3)
            out = out + self.conv3(out4)
        return out


class BRDNet(nn.Module):           # LU模块
    def __init__(self, in_nc=3, out_nc=3, nf=64):
        super(BRDNet, self).__init__()
        self.f_enc = nn.Sequential(
            nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.f_dec = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, out_nc, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.f_enc(x)
        x = self.f_dec(x)

        output_size = x.size()[2:]
        if output_size != input_size:
            pad_h = input_size[0] - output_size[0]
            pad_w = input_size[1] - output_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return x

class DenoiseBranch(nn.Module):
    def __init__(self):
        super(DenoiseBranch, self).__init__()
        self.denoise1 = FeatureBlock(in_channels=3, out_channels=3, layers=2)

    def forward(self, input):
        denoised_output = self.denoise1(input)
        return denoised_output



class Network(nn.Module):
    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)
        self.denoise_branch = DenoiseBranch()
        self._criterion = LossFunction()
        self.denoise = BRDNet(in_nc=3, out_nc=3, nf=64)
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input

        denoised_output = self.denoise_branch(input_op)

        for i in range(self.stage):
            inlist.append(input_op)
            i = self.enhance(input_op)
            resized_input_op = F.interpolate(input_op, size=i.shape[2:], mode='bilinear', align_corners=False)
            r = resized_input_op / i
            r = torch.clamp(r, 0, 1)
            att = self.calibrate(r)
            att = torch.nn.functional.interpolate(att, size=input_op.shape[2:], mode='bilinear', align_corners=False)
            input_op = input_op + att + denoised_output
            input_op = self.denoise(input_op)
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])
        return loss


class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self._criterion = LossFunction()


        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return i, r


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, r)
        return loss