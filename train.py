import os.path
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from network import *
from data import *

device = torch.device('cuda')
weight_path = 'params/unet.pth'
data_path = r'D:\Codes\DATA\VOCdevkit\VOC2012'
save_path = r'D:\Codes\myUnet\output'
if __name__ == '__main__':
    loader = DataLoader(myDataset(data_path), batch_size=4,shuffle=True)
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