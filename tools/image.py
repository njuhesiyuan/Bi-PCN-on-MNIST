import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


def visualize(img,path):

    img = img.detach().numpy()
    #img.reshape()
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.savefig(path)


