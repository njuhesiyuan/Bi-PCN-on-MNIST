from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset
import torch.nn.functional as F
import torch

from .image import visualize

def test_classify(model,dataloader):

    model.eval()
    correct = 0
    total = 0

    for input,target in dataloader:

        output = model.forward(input)

        pred = output.argmax(dim=1)
        true = target.argmax(dim=1)
        correct += (pred == true).sum().item()
        total += target.size(0)
    
    accuracy = 100 * correct / total

    print(f'分类任务准确率: {accuracy:.3f}%')


def test_generate(model,cl = F.one_hot(torch.tensor([1]),num_classes=10).float(),savepath = 'figures/generated_fig.png'):
    #print(cl)
    img = model.generate(cl)

    img = img.reshape((28,28))

    visualize(img,savepath)
