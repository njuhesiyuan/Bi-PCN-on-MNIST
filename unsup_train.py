from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset
import torchvision 
import torch
import copy
import torch.nn.functional as F

from models.unsup_BiPCN import  unsup_BiPCN
from tools.gen_data import generate_data


mini_train = True
saving = True


if __name__ == "__main__":

    
    train_loader,test_loader = generate_data(mini_train)



    model = unsup_BiPCN(output_dim=40)
    print(model.output_dim)
    model.unsupervised_train(train_loader)
    
    

    
    if saving == True:
        torch.save(model,'saved_models/unsupervised_BiPCN.pth')