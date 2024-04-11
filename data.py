import os
from utils import *
from torch.utils.data import Dataset
from torchvision import transforms

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

if __name__ == '__main__':
    data = myDataset('D:\Codes\DATA\VOCdevkit\VOC2012')
    print(data[0][0].shape)
    print(data[0][1].shape)
