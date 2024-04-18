it is recommanded to train your model on cloud server (if you have a RTX4090, ignore) 

load the code in lunchOnServer.py into jupyternotebook or use train.py to train the network, adjust the batch_size to avoid memory overflow. for 1070, use 4, 4090 use 20 (utilize learning rate warmup)

don't forget to create corresponding directory when launch on server
