from tools.test import test_classify,test_generate
from tools.gen_data import generate_data

import torch


mini_train = True

if __name__ == "__main__":

    train_loader,test_loader = generate_data(mini_train)

    model_path = 'saved_models/supervised_BiPCN.pth'
    model = torch.load(model_path,weights_only=False)
    #model.__init__()
    #print(model.input_dim)
    test_classify(model,test_loader)

    test_generate(model)