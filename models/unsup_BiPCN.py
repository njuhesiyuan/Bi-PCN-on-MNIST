# 无监督学习中Bi-PCN 的输出层为40维

from torch import nn
import torch
import torch.nn.functional as F
from torch import norm

from .BiPCN import BiPCN

class unsup_BiPCN(BiPCN):

    def train_infer(self,x,iters=100,lr=0.01):
        self.xf1 = x
        self.xb4 = x

        for i in range(iters):

            vb1 = F.relu(self.bc1(self.xb1))
            vb2 = F.relu(self.bc2(self.xb2))
            vb3 = F.sigmoid(self.bc3(self.xb3))
            vf1 = F.relu(self.fc1(self.xf1))
            vf2 = F.relu(self.fc2(self.xf2))
            vf3 = F.softmax(self.fc3(self.xf3),dim=1)

            E1 = (norm(vf1-self.xf2)+norm(vf2-self.xf3)+norm(vf3-self.xf4))/3
            E2 = (norm(vb1-self.xb2)+norm(vb2-self.xb3)+norm(vb3-self.xb4))/3
            E = self.alpha1*E1+self.alpha2*E2
            E.backward(retain_graph=True)

            self.xf2 = (self.xf2-self.xf2.grad*lr).detach().clone().requires_grad_(True)
            self.xf3 = (self.xf3-self.xf3.grad*lr).detach().clone().requires_grad_(True)
            self.xf4 = (self.xf4-self.xf4.grad*lr).detach().clone().requires_grad_(True)
            self.xb1 = (self.xb1-self.xb1.grad*lr).detach().clone().requires_grad_(True)
            self.xb2 = (self.xb2-self.xb2.grad*lr).detach().clone().requires_grad_(True)
            self.xb3 = (self.xb3-self.xb3.grad*lr).detach().clone().requires_grad_(True)

    def unsupervised_train_step(self,train_loader,lr=0.01):

        avg_E = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx>10:
                break
            self.train_infer(x)
            self.optimizer.zero_grad()

            vb1 = F.relu(self.bc1(self.xb1))
            vb2 = F.relu(self.bc2(self.xb2))
            vb3 = F.sigmoid(self.bc3(self.xb3))
            vf1 = F.relu(self.fc1(self.xf1))
            vf2 = F.relu(self.fc2(self.xf2))
            vf3 = F.softmax(self.fc3(self.xf3),dim = 1)

            E1 = (norm(vf1-self.xf2)+norm(vf2-self.xf3)+norm(vf3-self.xf4))/3
            E2 = (norm(vb1-self.xb2)+norm(vb2-self.xb3)+norm(vb3-self.xb4))/3
            E = self.alpha1*E1+self.alpha2*E2
            E.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            avg_E += E
        
        avg_E = avg_E/(batch_idx+1)
        return avg_E
        
    def unsupervised_train(self,train_loader,iters=100,lr = 0.01):

        for i in range(iters):
            #print(i)
            E = self.unsupervised_train_step(train_loader,lr)
            if (i+1)%10 ==0:
                print("第{}轮训练损失值为{:.4f}".format(i+1,E))

