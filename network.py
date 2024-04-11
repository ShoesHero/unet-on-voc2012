import torch
from torch import nn

class Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class DownSampling(nn.Module):
    def __init__(self,channel):
        super(DownSampling,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class UpSampling(nn.Module):
    def __init__(self,channel):
        super(UpSampling,self).__init__()
        self.layer = nn.Conv2d(channel, channel//2,1,1)
    def forward(self,x,y):
        up = nn.functional.interpolate(x,scale_factor=2,mode='nearest')
        out = self.layer(up)
        return torch.cat((y,out),dim=1)

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.c1 = Conv(3,64)
        self.d1 = DownSampling(64)
        self.c2 = Conv(64,128)
        self.d2 = DownSampling(128)
        self.c3 = Conv(128,256)
        self.d3 = DownSampling(256)
        self.c4 = Conv(256,512)
        self.d4 = DownSampling(512)
        self.c5 = Conv(512,1024)
        self.u1 = UpSampling(1024)
        self.c6 = Conv(1024,512)
        self.u2 = UpSampling(512)
        self.c7 = Conv(512, 256)
        self.u3 = UpSampling(256)
        self.c8 = Conv(256, 128)
        self.u4 = UpSampling(128)
        self.c9 = Conv(128, 64)
        self.out = nn.Conv2d(64,3,3,1,1)
        self.Th = nn.Sigmoid()

    def forward(self,x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        R6 = self.c6(self.u1(R5,R4))
        R7 = self.c7(self.u2(R6,R3))
        R8 = self.c8(self.u3(R7,R2))
        R9 = self.c9(self.u4(R8,R1))
        return self.Th(self.out(R9))

if __name__ == '__main__':
    x = torch.randn(2,3,256,256)
    unet = Unet()
    print(unet(x).shape)

