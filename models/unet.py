import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, padding_mode='reflect', down=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, stride=2 if down else 1,
                      kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, stride=1, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, down=True)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, conv1x1_ich, conv1x1_och=4):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv(in_channels+conv1x1_och, out_channels, mid_channels=out_channels)
        self.in_c = in_channels
        self.out_c = out_channels

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(conv1x1_ich, conv1x1_och, kernel_size=1, stride=1),
            nn.BatchNorm2d(conv1x1_och),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.conv1x1(x2)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv(x))


class UNet(nn.Module):
    def __init__(self, num_channels_down=None, num_channels_up=None, in_channels=3, out_channels=3, conv1x1_och=4):
        super(UNet, self).__init__()
        if num_channels_down is None:
            num_channels_down = [16, 32, 64, 128, 128]
        if num_channels_up is None:
            num_channels_up = [16, 32, 64, 128, 128]

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down1 = (DoubleConv(in_channels, num_channels_down[0], down=True))
        self.down2 = (Down(num_channels_down[0], num_channels_down[1]))
        self.down3 = (Down(num_channels_down[1], num_channels_down[2]))
        self.down4 = (Down(num_channels_down[2], num_channels_down[3]))
        self.down5 = (Down(num_channels_down[3], num_channels_down[4]))
        self.up0 = (Up(num_channels_up[4], num_channels_up[4], num_channels_up[3], conv1x1_och))
        self.up1 = (Up(num_channels_up[4], num_channels_up[3], num_channels_up[2], conv1x1_och))
        self.up2 = (Up(num_channels_up[3], num_channels_up[2], num_channels_up[1], conv1x1_och))
        self.up3 = (Up(num_channels_up[2], num_channels_up[1], num_channels_up[0], conv1x1_och))
        self.up4 = (Up(num_channels_up[1], num_channels_up[0], in_channels, conv1x1_och))
        self.rgb = (ToRGB(num_channels_up[0], out_channels))

    def forward(self, x):
        x0 = x
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up0(x5, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        y = self.rgb(x)
        return y
