import torch 
from torch import nn            
import torch.nn.functional as F         


class CTP_Predictor(nn.Module):
    
    def __init__(self,latent_dim,dropout=0.3):
        super(CTP_Predictor,self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim,1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Linear(1024,978)
    
    def forward(self,latents):
        #latents [batch,latents]
        preds = self.predictor(latents)
        outputs = self.output(preds)
        return outputs


class CTP_Predictor_resnet(nn.Module):
    #在dleps中使用残差连接
    def __init__(self,latent_dim,output_dim,dropout=0.3):
        super(CTP_Predictor_resnet,self).__init__()
        
        self.linear1 = nn.Linear(latent_dim,1024 )
        self.acti1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.block1 = ResidueBlock(input_dim=1024,output_dim=1024,dropout=dropout)
        self.block2 = ResidueBlock(input_dim=1024,output_dim=1024,dropout=dropout)
        self.block3 = ResidueBlock(input_dim=1024,output_dim=1024,dropout=dropout)
        self.linear2 = nn.Linear(1024,1024)
        self.acti2 = nn.Tanh()
        self.drop2 = nn.Dropout(dropout)
    
        self.output = nn.Linear(1024,output_dim)
        
    def forward(self,latents):
        out = self.drop1(self.acti1(self.linear1(latents)))
        
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        
        out = self.drop2(self.acti2(self.linear2(out)))
        output = self.output(out)
        return output
        

class ResidueBlock(nn.Module):
    def __init__(self,input_dim,output_dim,dropout):
        super(ResidueBlock,self).__init__()
        
        self.linear = nn.Linear(input_dim,output_dim)
        #self.bn = nn.BatchNorm1d(output_dim)
        self.activate = F.relu
        self.dropout = nn.Dropout(dropout)
        #
        #self.bn = nn.BatchNorm1d(output_dim)
    
    def forward(self,x):
        out = self.dropout(self.activate(self.linear(x)))
        return x + out