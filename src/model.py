import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from protein_net import *
from drug_net import *

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()

    def forward(self, v):
        v_out = self.linear(v)
        v_out = self.dropout(v_out)
        v_out = self.act(v_out)
        return v_out




class MoaNet(nn.Module):
    
    def __init__(self,**config):
        super(MoaNet,self).__init__()
        self.config = config
        self.use_drugsig = self.config["use_drugsig"]
        self.drug_model_type = self.config["drug_model_type"]
        protein_model_type = self.config["protein_model_type"]
        signature_type = self.config["signature_type"]
        
        if self.drug_model_type == "kpgt_embedding":
            
            self.model_drug = nn.Sequential(
                MLP(2304,1024),
                MLP(1024,256)
            )
        
        elif self.drug_model_type == "MDTips":
            self.model_drug = AttentiveFP_Model(
                in_channels=45,
                hidden_channels=256,
                num_layers=3,
                edge_dim=7,
                out_channels=64,num_timesteps=2,
                predictor_dim=256
                
            )
        
        if protein_model_type == "supervised":
            self.model_protein = prot_transformer()
        elif protein_model_type =="esm2":
            self.model_protein = nn.Sequential(MLP(1280,1024),
                                               MLP(1024,256))
        
        self.p_sig = nn.Sequential(MLP(978,1024),MLP(1024,256))
        if self.use_drugsig:
            self.d_sig = nn.Sequential(MLP(978,512),MLP(512,256))
            self.D_pro = MLP(512,256)
        
        if self.config["protein_model_type"] == "supervised":
            self.E_pro = MLP(320,256)
        else:
            self.E_pro = MLP(512,256)
        self.layernorm = LayerNorm(256)
        self.dropout = nn.Dropout(0.15)
        
        self.hidden_dims = [1024,1024,512]
        layer_size = len(self.hidden_dims) + 1
        
        dims = [512] + self.hidden_dims + [1]
        
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])
    
    def forward(self,data):
        
        if self.use_drugsig:
            if self.config["protein_model_type"] =="supervised":
                d_s = self.model_drug(data[0]) #[batch,256]
                d_expr = self.d_sig(data[1])
                p_s = self.model_protein(data[2],data[3]) #[batch,64]
                p_expr = self.p_sig(data[4])
                v_s = torch.cat((p_expr, p_s), 1)
                d_s = torch.cat((d_s,d_expr),1)
                d_s = self.D_pro(d_s)
                v_s = self.E_pro(v_s)
                
                v = torch.cat((d_s,v_s),dim=1) #[batch,512]
                
                for i,l in enumerate(self.predictor):
                    if i == (len(self.predictor) -1 ):
                        v = l(v)
                    else:
                        v = F.relu(self.dropout(l(v)))
                
                return v  #[batch,512]  
            
            elif self.config["protein_model_type"] == "esm2":
                    d_s = self.model_drug(data[0])
                    d_expr = self.d_sig(data[1])
                    p_s = self.model_protein(data[2])
                    p_expr = self.p_sig(data[3])
                    v_s = torch.cat((p_expr,p_s),dim=1)
                    d_s = torch.cat((d_s,d_expr),1)
                    d_s = self.D_pro(d_s)
                    v_s = self.E_pro(v_s)
                    
                    v = torch.cat((d_s,v_s),dim=1)
                    
                    for i,l in enumerate(self.predictor):
                        if i == (len(self.predictor) -1 ):
                            v = l(v)
                        else:
                            v = F.relu(self.dropout(l(v)))
                    
                    return v                               
        else:
            if self.config["protein_model_type"] =="supervised":
                d_s = self.model_drug(data[0]) #[batch,256]
                p_s = self.model_protein(data[1],data[2]) #[batch,64]
                p_expr = self.p_sig(data[3])
                v_s = torch.cat((p_expr, p_s), 1)
                
                v_s = self.E_pro(v_s)
                
                v = torch.cat((d_s,v_s),dim=1) #[batch,512]
                
                for i,l in enumerate(self.predictor):
                    if i == (len(self.predictor) -1 ):
                        v = l(v)
                    else:
                        v = F.relu(self.dropout(l(v)))
                
                return v  #[batch,512]
            
            elif self.config["protein_model_type"] == "esm2":
                d_s = self.model_drug(data[0])
                p_s = self.model_protein(data[1])
                p_expr = self.p_sig(data[2])
                v_s = torch.cat((p_expr,p_s),dim=1)
                
                v_s = self.E_pro(v_s)
                
                v = torch.cat((d_s,v_s),dim=1)
                
                for i,l in enumerate(self.predictor):
                    if i == (len(self.predictor) -1 ):
                        v = l(v)
                    else:
                        v = F.relu(self.dropout(l(v)))
                
                return v                   
            
            elif self.config["protein_model_type"] == "protein_cnn":
                d_s = self.model_drug(data[0])
                p_s = self.model_protein(data[1])
                p_expr = self.p_sig(data[2])
                v_s = torch.cat((p_expr,p_s),dim=1)
                
                v_s = self.E_pro(v_s)
                
                v = torch.cat((d_s,v_s),dim=1)
                
                for i,l in enumerate(self.predictor):
                    if i == (len(self.predictor) -1 ):
                        v = l(v)
                    else:
                        v = F.relu(self.dropout(l(v)))
                
                return v                   