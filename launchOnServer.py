import os

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import torch
from torch import nn

import os.path
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

device = torch.device('cuda')
weight_path = r'/root/autodl-tmp/myUnet/params/unet.pth'
data_path = r'/root/autodl-tmp/VOC2012'
save_path = r'/root/autodl-tmp/myUnet/output'
if __name__ == '__main__':
    loader = DataLoader(myDataset(data_path), batch_size=7,shuffle=True)
    unet = Unet().to(device)
    if os.path.exists(weight_path):
        unet.load_state_dict(torch.load(weight_path))
        print('successfully loaded weight')
    else:
        print('no weight loaded')
    opt = optim.Adam(unet.parameters())
    loss_func = nn.BCELoss()

    epoch = 1
    while True:
        for i, (image, seg_image) in enumerate(loader):
            image,seg_image = image.to(device), seg_image.to(device)
            out_image = unet(image)
            train_loss = loss_func(out_image, seg_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%10 == 0:
                print(f'{epoch}-{i}-train loss===>{train_loss.item()}')

            if i%100 == 0:
                torch.save(unet.state_dict(),weight_path)

                _image = image[0]
                _seg_image = seg_image[0]
                _out_image = out_image[0]

                img = torch.stack([_image,_seg_image,_out_image],dim=0)
                save_image(img,f'{save_path}/{i}.png')

        epoch += 1

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

def keep_image_size_open(path,size=(256,256)):
    img = Image.open(path)
    maxLength = max(img.size)
    mask = Image.new('RGB',(maxLength,maxLength),(0,0,0))
    mask.paste(img,(0,0))
    mask = mask.resize(size)
    return mask

transform = transforms.Compose(
    [transforms.ToTensor()]
)


class myDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.name = os.listdir(os.path.join(path,'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        seg_name =  self.name[index]  #png
        seg_path = os.path.join(self.path, 'SegmentationClass', seg_name)
        image_path = os.path.join(self.path, 'JPEGImages', seg_name.replace('png','jpg'))
        image = keep_image_size_open(image_path)
        seg_image = keep_image_size_open(seg_path)
        return transform(image), transform(seg_image)

