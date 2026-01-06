from tools.test import test_classify,test_generate
from tools.gen_data import generate_data

import torch
import torch.nn.functional as F

mini_train = True

if __name__ == "__main__":

    train_loader,test_loader = generate_data(mini_train)

    model_path = 'saved_models/unsupervised_BiPCN.pth'
    model = torch.load(model_path,weights_only=False)
    #model.__init__()
    #test_classify(model,test_loader)

    test_generate(model,cl = F.one_hot(torch.tensor([1]),num_classes=40).float(),savepath= 'figures/unsup_generated_figure.png')