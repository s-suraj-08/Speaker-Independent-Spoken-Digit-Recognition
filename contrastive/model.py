import torch.nn as nn
import torch

class contrastiveModel(nn.Module):
    def __init__(self, in_size, filters, out_size):
        '''
        in_size: is same as k
        '''
        super(contrastiveModel, self).__init__()
        
        layes=[]
        layes.append(nn.Conv1d(in_size, filters[0], 3, padding=1, padding_mode='reflect'))
        for i in range(len(filters)-1):
            layes.append( nn.Conv1d(filters[i], filters[i+1], 3, padding=1, padding_mode='reflect') )
            layes.append( nn.ReLU())
        self.layers = nn.Sequential(*layes)
        
        self.linears = nn.Sequential( nn.Linear(filters[-1], 256), nn.ReLU(), nn.Linear(256, 128) )
    
    def forward(self, x):
        x = x.permute(0,2,1).float()
        out = self.layers(x)
        out = torch.max(out, 2).values
        out = self.linears(out)
        return out