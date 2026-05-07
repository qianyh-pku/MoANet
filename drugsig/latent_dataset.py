import numpy as np                    
import pandas as pd     
import torch 
from torch import nn          
from torch.utils.data import Dataset,DataLoader

class LatentDataset(Dataset):
    #输入latent和对应的表达谱 供训练使用
    def __init__(self,latent_path,rna_path):
        self.latent_path = latent_path
        self.rna_path = rna_path
        self.latent = np.load(latent_path,allow_pickle=True).astype(np.float32)
        self.gene_signature = pd.read_csv(rna_path,index_col=0).to_numpy().astype(np.float32)
        #self.gene_signature = np.load(rna_path,allow_pickle=True)
        assert self.latent.shape[0] == self.gene_signature.shape[0]
        
    def __getitem__(self, index):
        latent = self.latent[index]
        gene = self.gene_signature[index]
        return torch.tensor(latent),torch.tensor(gene)
    
    def __len__(self):
        return self.latent.shape[0]


    def collate_fn(self,samples):
        latents = [s[0] for s in samples]
        genes = [s[1] for s in samples]
        mols = torch.stack(latents,dim=0)
        genes = torch.stack(genes,dim=0)
        
        return {"latents":mols,"gene_signatures":genes}


class PredictLatentDataset(Dataset):
    #只需要输入latent vector 用来推理使用
    def __init__(self,latent_path):
        self.latent_path = latent_path
        self.latent = np.load(latent_path,allow_pickle=True)
    
    def __getitem__(self,index):
        latent = self.latent[index]
        return torch.tensor(latent)
    
    def __len__(self):
        return self.latent.shape[0]
    
    def collate_fn(self,samples):
        latents = torch.stack(samples,dim=0)
        return latents
    
