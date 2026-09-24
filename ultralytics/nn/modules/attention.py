import torch.nn as nn


class GAM(nn.Module):
    def __init__(self, in_channels, out_channels=None, rate=4):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels) if out_channels is not None else in_channels
        inchannel_rate = int(in_channels / rate)

        # Channel attention (MLP after 3D permutation)
        self.linear1 = nn.Linear(in_channels, inchannel_rate)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inchannel_rate, in_channels)

        # Spatial attention
        self.conv1 = nn.Conv2d(in_channels, inchannel_rate, kernel_size=7, padding=3, padding_mode='replicate')
        self.conv2 = nn.Conv2d(inchannel_rate, out_channels, kernel_size=7, padding=3, padding_mode='replicate')
        self.norm1 = nn.BatchNorm2d(inchannel_rate)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.linear2(self.relu(self.linear1(x_permute))).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)  # B,C,H,W
        x = x * x_channel_att

        # Spatial attention
        x_spatial_att = self.relu(self.norm1(self.conv1(x)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(x_spatial_att)))
        out = x * x_spatial_att
        return out
