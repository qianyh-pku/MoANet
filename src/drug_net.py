import torch        
from torch import nn       
from torch_geometric.nn import GAT,GCN,GIN
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool,GlobalAttention,Set2Set
from torch_geometric.nn.models import AttentiveFP

class GAT_Model(nn.Module):
    
    def __init__(self,in_channels,hidden_channels,num_layers,out_channels,heads,predictor_dim,concat=False,dropout=0.3,**kwargs):
        
        super(GAT_Model,self).__init__()
        self.gnn = GAT(in_channels=in_channels,
                       hidden_channels=hidden_channels,
                       num_layers=num_layers,
                       out_channels=out_channels,
                       dropout=dropout,
                       v2=True,
                       heads=heads,
                       concat = concat,
                       **kwargs
        )
        self.pooling = Set2Set(in_channels=out_channels,processing_steps=2)
        self.transform = nn.Linear(out_channels*2,predictor_dim)
    
    def forward(self,graph):
        
        x,edge_index,edge_attr,batch = graph.x,graph.edge_index,graph.edge_attr,graph.batch   
        
        emb = self.gnn(x=x,edge_index=edge_index,batch=batch,edge_attr=edge_attr)
        emb = self.pooling(emb,batch)
        
        return self.transform(emb)


class GCN_Model(nn.Module):
    
    def __init__(self,in_channels,hidden_channels,num_layers,out_channels,predictor_dim,dropout=0.3,**kwargs):
        super(GCN_Model,self).__init__()
        
        self.gnn = GCN(in_channels=in_channels,
                       hidden_channels=hidden_channels,
                       num_layers=num_layers,
                       out_channels=out_channels,
                       dropout=dropout,
                       **kwargs 
                       )
        self.transform = nn.Linear(out_channels*2,predictor_dim)
        self.pooling = Set2Set(in_channels=out_channels,processing_steps=2)
    
    def forward(self,graph):
        x,edge_index,edge_attr,batch = graph.x,graph.edge_index,graph.edge_attr,graph.batch   
        emb = self.gnn(x=x,edge_index=edge_index,batch=batch,edge_attr=edge_attr)
        emb = self.pooling(emb,batch)
        return self.transform(emb)


class GIN_Model(nn.Module):
    
    def __init__(self,in_channels,hidden_channels,num_layers,out_channels,predictor_dim,dropout=0.3,**kwargs):
        super(GIN_Model,self).__init__()
        
        self.gnn = GIN(in_channels=in_channels,
                       hidden_channels=hidden_channels,
                       num_layers=num_layers,
                       out_channels=out_channels,
                       dropout=dropout,**kwargs)
        self.pooling = Set2Set(in_channels=out_channels,processing_steps=2)
        self.transform = nn.Linear(out_channels*2,predictor_dim)
    
    def forward(self,graph):
        x,edge_index,edge_attr,batch = graph.x,graph.edge_index,graph.edge_attr,graph.batch   
        emb = self.gnn(x=x,edge_index=edge_index,batch=batch,edge_attr=edge_attr)
        emb = self.pooling(emb,batch)
        return self.transform(emb)

class AttentiveFP_Model(nn.Module):
    #这个是MDTips论文中使用的方法
    def __init__(self,in_channels,hidden_channels,num_layers,edge_dim,out_channels,num_timesteps,predictor_dim,dropout=0.3,**kwargs):
        super(AttentiveFP_Model,self).__init__()
        self.gnn = AttentiveFP(in_channels=in_channels,
                               hidden_channels=hidden_channels,
                               out_channels=out_channels,
                               edge_dim=edge_dim,num_layers=num_layers,num_timesteps=num_timesteps,dropout=dropout,**kwargs)
        
        self.transform = nn.Linear(out_channels,predictor_dim)
        
    def forward(self,graph):
        x,edge_index,edge_attr,batch = graph.x,graph.edge_index,graph.edge_attr,graph.batch
        emb = self.gnn(x=x,edge_index=edge_index,batch=batch,edge_attr=edge_attr)
        return self.transform(emb)
        
        
        