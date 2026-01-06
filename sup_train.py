from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset
import torchvision 
import torch
import copy
import torch.nn.functional as F

from models.BiPCN import  BiPCN
from tools.gen_data import generate_data


mini_train = True
saving = True


if __name__ == "__main__":

    
    train_loader,test_loader = generate_data(mini_train)



    model = BiPCN()
    model.supervised_train(train_loader)
    
    

    
    if saving == True:
        torch.save(model,'saved_models/supervised_BiPCN.pth')


