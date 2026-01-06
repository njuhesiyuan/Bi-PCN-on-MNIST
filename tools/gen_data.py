import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset
import torch
import torch.nn.functional as F

def generate_data(mini_train = True):

    train_dataset = torchvision.datasets.MNIST(
        root = 'data',  
        train = True,       
        transform = torchvision.transforms.ToTensor(),   
        download=False
    )


    test_dataset = torchvision.datasets.MNIST(
        root = 'data',  
        train = False,       
        transform = torchvision.transforms.ToTensor(),  
        download=False
    )
    
    
    class minist(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.num_classes = 10

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            img = img.view(-1)
            label_onehot = F.one_hot(torch.tensor(label), num_classes=10).float()
            return img, label_onehot
    
    
    train_dataset = minist(train_dataset)
    test_dataset = minist(test_dataset)

    #print(train_dataset[0][0], train_dataset[0][1])

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64,shuffle=True)
    
    train_indice = list(range(640))
    test_indice = list(range(64))
    mini_train_dataset = Subset(train_dataset,train_indice)
    mini_test_dataset = Subset(test_dataset,test_indice)

    mini_train_loader = DataLoader(mini_train_dataset,batch_size=64,shuffle=True)
    mini_test_loader = DataLoader(mini_test_dataset, batch_size=64,shuffle=True)

    if mini_train == True:
        return mini_train_loader,mini_test_loader
    else:
        return train_loader,test_loader


