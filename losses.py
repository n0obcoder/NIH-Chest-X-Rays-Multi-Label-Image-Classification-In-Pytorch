import torch, sys, os, pdb
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    
    def __init__(self, device, gamma = 1.0):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = torch.tensor(gamma, dtype = torch.float32).to(device)
        self.eps = 1e-6
        
#         self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)
        
    def forward(self, input, target):
        
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').to(self.device)
#         BCE_loss = self.BCE_loss(input, target)
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss =  (1-pt)**self.gamma * BCE_loss
        
        return F_loss.mean() 

    # def forward(self, input, target):

    #     # input are not the probabilities, they are just the cnn out vector
    #     # input and target shape: (bs, n_classes)
    #     # sigmoid
    #     probs = torch.sigmoid(input)
    #     log_probs = -torch.log(probs)

    #     focal_loss = torch.sum(   torch.pow(1-probs + self.eps, self.gamma).mul(log_probs).mul(target)   , dim=1)
    #     # bce_loss = torch.sum(log_probs.mul(target), dim = 1)
        
    #     return focal_loss.mean() #, bce_loss

if __name__ == '__main__':
    inp = torch.tensor([[1., 0.95], 
                        [.9, 0.3], 
                        [0.6, 0.4]], requires_grad = True)
    target = torch.tensor([[1., 1], 
                        [1, 0], 
                        [0, 0]])

    print('inp\n',inp, '\n')
    print('target\n',target, '\n')

    print('inp.requires_grad:', inp.requires_grad, inp.shape)
    print('target.requires_grad:', target.requires_grad, target.shape)


    loss = FocalLoss(gamma = 2)

    focal_loss, bce_loss = loss(inp ,target)
    print('\nbce_loss',bce_loss, '\n')
    print('\nfocal_loss',focal_loss, '\n')
