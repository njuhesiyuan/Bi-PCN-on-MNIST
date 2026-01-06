import torchvision 

train_data = torchvision.datasets.MNIST(
    root = 'data',  
    train = True,       
    transform = torchvision.transforms.ToTensor(),   
    download=True
)


test_data = torchvision.datasets.MNIST(
    root = 'data',  
    train = False,       
    transform = torchvision.transforms.ToTensor(),  
    download=True
)