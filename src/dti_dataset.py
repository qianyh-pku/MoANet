import joblib
import pandas as pd
import numpy as np
import torch     
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import codecs
from subword_nmt.apply_bpe import BPE

from tqdm import tqdm
import logging
from protein_net import *

vocab_path = "data/ESPF/protein_codes_uniprot_2000.txt"
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv("data/ESPF/subword_units_map_uniprot_2000.csv")
idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25



def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)

def integer_label_protein(sequence, max_length=545):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding


# def get_protein_emb_esm2(seq,esm_model,esm_batch_converter,esm_alphabet,max_len=1024,per_token=False):
#     seq = seq.upper()
#     if len(seq) > max_len - 2:
#         seq = seq[: max_len - 2]
    
#     batch_labels, batch_strs, batch_tokens = esm_batch_converter(
#             [("sequence", seq)]
#         )
#     batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)
    
#     batch_tokens = batch_tokens.to(device)
    
#     with torch.no_grad():
#         results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
#     token_representations = results["representations"][33][0]
    
#     if per_token:
#         return token_representations

#     return token_representations.mean(0).detach().cpu()



def data_process(train,val,test,**config):
    
    train_pro = {}
    val_pro = {}
    test_pro = {}
    
    if config["drug_model_type"] in ["GAT","GIN","GCN","MDTips"]:
        print("loading smiles embedding from gnn_graph")
        d_graph_dict  = joblib.load("/home/user/qianyh/plifts/data/smiles_pretrained/d_graph_dict.pkl")
    
    elif config["drug_model_type"] == "kpgt_embedding":
        d_graph_dict = joblib.load("/home/user/qianyh/plifts/data/smiles_pretrained/kpgt_latent.pkl")
        
    elif config["drug_model_type"] == "agbt_embedding":
        d_graph_dict = joblib.load("/home/user/qianyh/plifts/data/smiles_pretrained/agbt_latent.pkl")
        
    elif config["drug_model_type"] == "unimol_embedding":
        d_graph_dict = joblib.load("/home/user/qianyh/plifts/data/smiles_pretrained/unimol_latent.pkl")
    
    elif config["drug_model_type"] == "molclr_embedding":
        d_graph_dict = joblib.load("/home/user/qianyh/plifts/data/smiles_pretrained/molclr_latent.pkl")
    
    elif config["drug_model_type"] == "molmcl_embedding":
        d_graph_dict = joblib.load("/home/user/qianyh/plifts/data/smiles_pretrained/molmcl_latent.pkl")
    
    elif config["drug_model_type"] == "chemberta_embedding":
        d_graph_dict = joblib.load("/home/user/qianyh/plifts/data/smiles_pretrained/chemberta_latent.pkl")
    
    elif config["drug_model_type"] == "ecfp_embedding":
        d_graph_dict = joblib.load("/home/user/qianyh/plifts/data/smiles_pretrained/ecfp_latent.pkl")
    
    elif config["drug_model_type"] == "molgt_embedding":
        d_graph_dict = joblib.load("/home/user/qianyh/plifts/data/smiles_pretrained/molgt_latent.pkl")
    
    df = pd.concat([train,val,test])
    smiles = list(set(df["smiles"]))
    proteins = list(set(df["seq"]))
    
    if config["signature_type"] == "gene2vec":
        oe_expr_df = pd.read_csv("/home/user/qianyh/plifts/data/pert_signature/oe_emb_512_consensus_frogs.csv",index_col=0)
        sh_expr_df = pd.read_csv("/home/user/qianyh/plifts/data/pert_signature/sh_emb_512_consensus_frogs.csv",index_col=0)
    
    elif config["signature_type"] =="landmark":
        oe_expr_df = pd.read_csv("/home/user/qianyh/plifts/data/pert_signature/oe_sig_level5_consensus_978.csv",index_col=0)
        sh_expr_df = pd.read_csv("/home/user/qianyh/plifts/data/pert_signature/sh_sig_level5_consensus_978.csv",index_col=0)
        
    # d_graph_dict = {}
    for smi in smiles:
        assert smi in d_graph_dict.keys()
    
    if config["protein_model_type"] == "supervised":
        print("prepare protein structure")
        passed_protein_graph = []
        passed_protein = []
        
        for prot in tqdm(proteins):
            prot_ids,prot_mask = protein2emb_encoder(prot)
            if prot is not None:
                passed_protein.append(prot)
                passed_protein_graph.append((prot_ids,prot_mask))
        
        assert len(passed_protein_graph) == len(passed_protein)
        p_graph_dict = dict(zip(passed_protein,passed_protein_graph))
    
    if config["protein_model_type"] == "protein_cnn":
        print("prepare protein structure")
        passed_protein_graph = []
        passed_protein = []
        
        for prot in tqdm(proteins):
            prot_ids = integer_label_protein(prot)
            if prot is not None:
                passed_protein.append(prot)
                passed_protein_graph.append(prot_ids)
        
        assert len(passed_protein_graph) == len(passed_protein)
        p_graph_dict = dict(zip(passed_protein,passed_protein_graph))
    
    elif config["protein_model_type"] =="esm2":
        import esm  
        #导入模型和tokenizer
        esm_model,esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        esm_batch_converter = esm_alphabet.get_batch_converter()
        esm_model.to(device)
        
        print("prepare protein embeddings...")
        passed_protein = []
        passed_protein_graph = []
        
        for prot in tqdm(proteins):
            prot_emb = get_protein_emb_esm2(prot,esm_model=esm_model,esm_batch_converter=esm_batch_converter,esm_alphabet=esm_alphabet)
            if prot_emb is not None:
                passed_protein.append(prot)
                passed_protein_graph.append(prot_emb)
        
    elif config["protein_model_type"] == "prott5":
        seq2prott5 = joblib.load("/home/user/qianyh/plifts_v2/data/seq2prott5.pkl")
        passed_protein = []
        passed_protein_graph = []
        
        for prot in tqdm(proteins):
            passed_protein.append(prot)
            passed_protein_graph.append(seq2prott5[prot])
        
    assert len(passed_protein_graph) == len(passed_protein)
    p_graph_dict = dict(zip(passed_protein,passed_protein_graph))
    
    keep_D_item = pd.DataFrame(smiles,columns=["smiles"])
    keep_P_item = pd.DataFrame(passed_protein,columns=["seq"])
    
    df = pd.merge(df,keep_D_item,on='smiles')
    df = pd.merge(df,keep_P_item,on="seq")
    
    train = pd.merge(train,keep_D_item,on="smiles")
    train_pro_df = pd.merge(train,keep_P_item,on="seq")
    train_pro["df"] = train_pro_df
    
    val = pd.merge(val,keep_D_item,on="smiles")
    val_pro_df = pd.merge(val,keep_P_item,on="seq")
    val_pro["df"] = val_pro_df
    
    test = pd.merge(test,keep_D_item,on="smiles")
    test_pro_df = pd.merge(test,keep_P_item,on="seq")
    test_pro["df"] = test_pro_df
    
    train_p_expr = []
    
    for i in tqdm(range(len(train_pro["df"]))):
        gene_symbol = train_pro["df"].iloc[i,2]

        if train_pro["df"].iloc[i,3] == 1:
            train_p_expr.append(np.array(oe_expr_df.loc[gene_symbol,:]))
        else:
            train_p_expr.append(np.array(sh_expr_df.loc[gene_symbol,:]))
    
    val_p_expr = []
    
    for i in tqdm(range(len(val_pro["df"]))):
        gene_symbol = val_pro["df"].iloc[i,2]
        if val_pro["df"].iloc[i,3] == 1:
            val_p_expr.append(np.array(oe_expr_df.loc[gene_symbol,:]))
        else:
            val_p_expr.append(np.array(sh_expr_df.loc[gene_symbol,:]))
    
    test_p_expr = []
    
    for i in tqdm(range(len(test_pro["df"]))):
        gene_symbol = test_pro["df"].iloc[i,2]
        if test_pro["df"].iloc[i,3] == 1:
            test_p_expr.append(np.array(oe_expr_df.loc[gene_symbol]))
        else:
            test_p_expr.append(np.array(sh_expr_df.loc[gene_symbol]))
    
    train_pro["P_expr"] = train_p_expr
    val_pro["P_expr"] = val_p_expr
    test_pro["P_expr"] = test_p_expr
    
    print('prepare final D_Structure')
    train_pro["D_structure"] = [d_graph_dict[i] for i in train_pro["df"]["smiles"]]
    val_pro["D_structure"] = [d_graph_dict[i] for i in val_pro["df"]["smiles"]]
    test_pro["D_structure"] = [d_graph_dict[i] for i in test_pro["df"]["smiles"]]
    
    print('prepare final P_graph')
    train_pro["P_structure"] = [p_graph_dict[i] for i in train_pro["df"]["seq"]]        
    val_pro["P_structure"] = [p_graph_dict[i] for i in val_pro["df"]["seq"]]    
    test_pro["P_structure"] = [p_graph_dict[i] for i in test_pro["df"]["seq"]]    
    
    train_pro['Label'] = torch.tensor(train_pro['df']['label'].values).to(torch.float32)
    val_pro['Label'] = torch.tensor(val_pro['df']['label'].values).to(torch.float32)
    test_pro['Label'] = torch.tensor(test_pro['df']['label'].values).to(torch.float32)
    
    return train_pro,val_pro,test_pro

def data_process_for_repurpose(smiles_list,d_graph_dict,protein_sequence,target_name,ups,**config):
    passed_smiles = []
    passed_graph = []
    
    for smi in smiles_list:
        if smi in d_graph_dict.keys():
            passed_smiles.append(smi)
            passed_graph.append(d_graph_dict[smi])
    
    if config["signature_type"] == "gene2vec":
        oe_expr_df = pd.read_csv("/home/user/qianyh/plifts/data/pert_signature/oe_emb_512_consensus_frogs.csv",index_col=0)
        sh_expr_df = pd.read_csv("/home/user/qianyh/plifts/data/pert_signature/sh_emb_512_consensus_frogs.csv",index_col=0)
    
    elif config["signature_type"] =="landmark":
        if ups:
            target_signature = pd.read_csv("/home/user/qianyh/plifts/data/pert_signature/oe_sig_level5_consensus_978.csv",index_col=0)
            target_signature = np.array(target_signature.loc[target_name,:]).astype(np.float32)
        else:
            target_signature = pd.read_csv("/home/user/qianyh/plifts/data/pert_signature/sh_sig_level5_consensus_978.csv",index_col=0)
            target_signature = np.array(target_signature.loc[target_name,:]).astype(np.float32)
    
    if config["protein_model_type"] == "supervised":
        prot_idx,prot_mask = protein2emb_encoder(protein_sequence)
        return passed_smiles,passed_graph,(prot_idx,prot_mask),target_signature
    
    elif config["protein_model_type"] =="esm2":
        p_graph_dict = joblib.load("/home/user/qianyh/ai_mol_target/data/merge_data/protein_esm_embed.pkl")
        p_s = p_graph_dict[protein_sequence]
        return passed_smiles,passed_graph,p_s,target_signature
        

class DTI_Dataset(Dataset):
    
    def __init__(self,data_df,**config):
        super(DTI_Dataset,self).__init__()
        
        self.data_df = data_df
        self.config = config
        
    def __len__(self):
        return len(self.data_df["df"])
    
    def __getitem__(self, index):
        if self.config["protein_model_type"] == "supervised":
            d_s = self.data_df["D_structure"][index]
            p_s,mask = self.data_df["P_structure"][index]
            p_expr = self.data_df["P_expr"][index]
            y = self.data_df["Label"][index]
            
            if self.config["drug_model_type"] in ["kpgt_embedding","agbt_embedding","molformer_embedding","molclr_embedding","ecfp_embedding","molmcl_embedding","unimol_embedding","chemberta_embedding","molgt_embedding"]:
                return torch.from_numpy(d_s),torch.from_numpy(p_s),torch.from_numpy(mask),torch.from_numpy(p_expr),y 
            return d_s,torch.from_numpy(p_s),torch.from_numpy(mask),torch.from_numpy(p_expr),y 
        
        elif self.config["protein_model_type"] == "protein_cnn":
            d_s = self.data_df["D_structure"][index]
            p_s = self.data_df["P_structure"][index]
            p_expr = self.data_df["P_expr"][index]
            y = self.data_df["Label"][index]
            if self.config["drug_model_type"] in ["kpgt_embedding","agbt_embedding","molformer_embedding","molclr_embedding","ecfp_embedding","molmcl_embedding","unimol_embedding","chemberta_embedding","molgt_embedding"]:
                return torch.from_numpy(d_s),torch.from_numpy(p_s).to(torch.float32),torch.from_numpy(p_expr),y 
            return d_s,torch.from_numpy(p_s).to(torch.float32),torch.from_numpy(p_expr),y 
        
        elif self.config["protein_model_type"] in ["esm2","prott5"]:
            d_s = self.data_df["D_structure"][index]
            p_s = self.data_df["P_structure"][index]
            p_expr = self.data_df["P_expr"][index]
            y = self.data_df["Label"][index]
            if self.config["drug_model_type"] in ["kpgt_embedding","agbt_embedding","molformer_embedding","molclr_embedding","ecfp_embedding","molmcl_embedding","unimol_embedding","chemberta_embedding","molgt_embedding"]:
                return torch.from_numpy(d_s),p_s,torch.from_numpy(p_expr),y 
            return d_s,p_s,torch.from_numpy(p_expr),y 
        
    def collate_fn(self,x):
        if self.config["protein_model_type"] == "supervised":
            d_s,p_s,mask,p_expr,y = zip(*x)
            if self.config["drug_model_type"] in ["kpgt_embedding","agbt_embedding","molformer_embedding","molclr_embedding","ecfp_embedding","molmcl_embedding","unimol_embedding","chemberta_embedding","molgt_embedding"]:
                return torch.stack(d_s,dim=0).to(torch.float32),torch.stack(p_s,dim=0),torch.stack(mask,dim=0),torch.stack(p_expr,dim=0).to(torch.float32),torch.stack(y,dim=0)
            else:
                d_s = Batch.from_data_list(d_s)
                
            return d_s,torch.stack(p_s,dim=0),torch.stack(mask,dim=0),torch.stack(p_expr,dim=0),torch.stack(y,dim=0)
        
        elif self.config["protein_model_type"] == "protein_cnn":
            d_s,p_s,p_expr,y = zip(*x)
            if self.config["drug_model_type"] in ["kpgt_embedding","agbt_embedding","molformer_embedding","molclr_embedding","ecfp_embedding","molmcl_embedding","unimol_embedding","chemberta_embedding","molgt_embedding"]:
                return torch.stack(d_s,dim=0).to(torch.float32),torch.stack(p_s,dim=0),torch.stack(p_expr,dim=0).to(torch.float32),torch.stack(y,dim=0)
            else:
                d_s = Batch.from_data_list(d_s)
                
            return d_s,torch.stack(p_s,dim=0),torch.stack(p_expr,dim=0),torch.stack(y,dim=0)
        
        elif self.config["protein_model_type"] in ["esm2","prott5"]:
            d_s,p_s,p_expr,y = zip(*x)
            if self.config["drug_model_type"] in ["kpgt_embedding","agbt_embedding","molformer_embedding","molclr_embedding","ecfp_embedding","molmcl_embedding","unimol_embedding","chemberta_embedding","molgt_embedding"]:
                return torch.stack(d_s,dim=0).to(torch.float32),torch.stack(p_s,dim=0),torch.stack(p_expr,dim=0).to(torch.float32),torch.stack(y,dim=0)
            else:
                d_s = Batch.from_data_list(d_s)
                
            return d_s,torch.stack(p_s,dim=0),torch.stack(p_expr,dim=0),torch.stack(y,dim=0)

class DTI_Dataset_For_Drug_Embedding(Dataset):
    
    def __init__(self,d_s):
        self.d_s = d_s         
    
    def __getitem__(self,index):
        return torch.from_numpy(self.d_s[index]).to(torch.float32)
    
    def __len__(self):
        return self.d_s.shape[0]

class DTI_Dataset_For_Repurpose(Dataset):
    
    def __init__(self,smiles_graph,p_s,p_mask,target_signature,**config):
        self.smiles_graph = smiles_graph
        self.p_s = p_s
        self.p_mask = p_mask 
        self.target_signature = target_signature
        self.config = config 
    
    def __len__(self):
        return len(self.smiles_graph)
    
    def __getitem__(self,index):
        d_s = self.smiles_graph[index]
        return torch.from_numpy(d_s).to(torch.float32),torch.from_numpy(self.p_s),torch.from_numpy(self.p_mask),torch.from_numpy(self.target_signature).to(torch.float32)
    
    def collate_fn(self,x):
        d_s,p_s,p_mask,p_expr = zip(*x)
        return torch.stack(d_s,dim=0),torch.stack(p_s,dim=0),torch.stack(p_mask,dim=0),torch.stack(p_expr,dim=0)
        
    

if __name__ == "__main__":
    
    kind = "scaffold"
    seed = 23
    
    config = {
    "drug_model_type":"kpgt_embedding",
    "protein_model_type":"esm2",
    "signature_type":"landmark",
    'result_folder':f"/home/user/qianyh/plifts/ckpts/{kind}/seed_{seed}",
    'LR':1e-3,
    'decay':0,
    'batch_size':256,
    'epochs':50,
    "seed":23,
    "early_stop":5
}
    
    
    
    train = pd.read_csv(f"/home/user/qianyh/plifts_submit_20241026/data/split_seed_23/{kind}/train.csv")
    val = pd.read_csv(f"/home/user/qianyh/plifts_submit_20241026/data/split_seed_23/{kind}/valid.csv")
    test = pd.read_csv(f"/home/user/qianyh/plifts_submit_20241026/data/split_seed_23/{kind}/test.csv")
    
    train, val, test = data_process(train,val,test,**config)