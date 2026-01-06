from torch import nn
import torch
import torch.nn.functional as F
from torch import norm

class BiPCN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10,batch_size = 64):
        super(BiPCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.xf1 = torch.rand((batch_size,input_dim),requires_grad=True)
        self.xf2 = torch.rand((batch_size,hidden_dim),requires_grad=True)
        self.xf3 = torch.rand((batch_size,hidden_dim),requires_grad=True)
        self.xf4 = torch.rand((batch_size,output_dim),requires_grad=True)

        self.xb4 = torch.rand((batch_size,input_dim),requires_grad=True)
        self.xb3 = torch.rand((batch_size,hidden_dim),requires_grad=True)
        self.xb2 = torch.rand((batch_size,hidden_dim),requires_grad=True)
        self.xb1 = torch.rand((batch_size,output_dim),requires_grad=True)


        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.bc1 = nn.Linear(output_dim, hidden_dim)
        self.bc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bc3 = nn.Linear(hidden_dim,input_dim)


        self.alpha1 = 0.7
        self.alpha2 = 0.3
        
        self.optimizer =  torch.optim.Adam(self.parameters(), lr = 0.01)
        '''
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        '''

    
    def forward(self, x,iters=100,lr = 0.01):

        l = x.shape[0]

        self.xf1 = x
        
        self.xf2 = torch.rand((l,self.hidden_dim),requires_grad=True)
        self.xf3 = torch.rand((l,self.hidden_dim),requires_grad=True)
        self.xf4 = torch.rand((l,self.output_dim),requires_grad=True)


        for i in range(iters):

            vf1 = F.relu(self.fc1(self.xf1))
            vf2 = F.relu(self.fc2(self.xf2))
            vf3 = F.softmax(self.fc3(self.xf3),dim=1)

            E = (norm(vf1-self.xf2)+norm(vf2-self.xf3)+norm(vf3-self.xf4))/3
            E.backward()

            self.xf2 = (self.xf2-self.xf2.grad*lr).detach().clone().requires_grad_(True)
            self.xf3 = (self.xf3-self.xf3.grad*lr).detach().clone().requires_grad_(True)
            self.xf4 = (self.xf4-self.xf4.grad*lr).detach().clone().requires_grad_(True)
        
        indices = torch.argmax(self.xf4, dim=1)
        one_hot = F.one_hot(indices, num_classes=self.output_dim)

        return one_hot
    
    def generate(self,y,iters=100,lr=0.01):

        l = y.shape[0]

        self.xb1 = y
        self.xb4 = torch.rand((l,self.input_dim),requires_grad=True)
        self.xb3 = torch.rand((l,self.hidden_dim),requires_grad=True)
        self.xb2 = torch.rand((l,self.hidden_dim),requires_grad=True)

        for i in range(iters):

            vb1 = F.relu(self.bc1(self.xb1))
            vb2 = F.relu(self.bc2(self.xb2))
            vb3 = F.sigmoid(self.bc3(self.xb3))

            E = (norm(vb1-self.xb2)+norm(vb2-self.xb3)+norm(vb3-self.xb4))/3
            E.backward()

            self.xb2 = (self.xb2-self.xb2.grad*lr).detach().clone().requires_grad_(True)
            self.xb3 = (self.xb3-self.xb3.grad*lr).detach().clone().requires_grad_(True)
            self.xb4 = (self.xb4-self.xb4.grad*lr).detach().clone().requires_grad_(True)
        
        return self.xb4
    
    def train_infer(self,x,y,iters=100,lr=0.01):

        self.xf1 = x
        self.xb4 = x
        self.xf4 = y
        self.xb1 = y

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
            self.xb2 = (self.xb2-self.xb2.grad*lr).detach().clone().requires_grad_(True)
            self.xb3 = (self.xb3-self.xb3.grad*lr).detach().clone().requires_grad_(True)




    def supervised_train_step(self,train_loader,lr=0.01):
        #print('train')
        avg_E = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx>10:
                break
            self.train_infer(x,y)
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
    
    def supervised_train(self,train_loader,iters=100,lr = 0.01):

        for i in range(iters):
            #print(i)
            E = self.supervised_train_step(train_loader,lr)
            if (i+1)%10 ==0:
                print("第{}轮训练损失值为{:.4f}".format(i+1,E))







        
        

