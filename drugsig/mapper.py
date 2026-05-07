import torch 
from torch import nn
import torch.nn.functional as F             
from torch.utils.data import Dataset,DataLoader

class BasicDataset(Dataset):
    
    def __init__(self,X,y):
        super(Dataset,self).__init__()
        
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    
    def __getitem__(self,index):
        return self.X[index],self.y[index]
    
    def __len__(self):
        return self.X.shape[0]
    
class GeneMapper(nn.Module):
    
    def __init__(self,input_dim,output_dim,dropout=0.3):
        super(GeneMapper,self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096,8192),
            nn.BatchNorm1d(8192),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(8192,output_dim)
        )
    
    def forward(self,x):
        output = self.model(x)
        return output

