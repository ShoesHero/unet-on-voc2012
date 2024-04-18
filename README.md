it is recommanded to train your model on cloud server (if you have a RTX4090, ignore) 

load the code in lunchOnServer.py into jupyternotebook or use train.py to train the network, adjust the batch_size to avoid memory overflow. for 1070, use 4, 4090 use 20 (utilize learning rate warmup)

don't forget to create corresponding directory when launch on server

training result looks like this:

![100](https://github.com/ShoesHero/unet/assets/113640926/aa09b7c1-24fa-4f2c-851b-fdf16a56f2dc)
![200](https://github.com/ShoesHero/unet/assets/113640926/868c2d30-c2d8-4bad-8a88-5f1fb7337da8)
![200 (1)](https://github.com/ShoesHero/unet/assets/113640926/3373da23-2efd-41ce-88f3-71d74c169567)
