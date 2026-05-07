import os,sys
import numpy as np
import pandas as pd
import pickle 


class GeneMapper:
    #根据预测的978个基因，map到12328个基因中，同时筛选高表达基因
    
    def __init__(self,setmean=False,reverse=True,base=-2,up_name=None,down_name=None,save_exp=None,geneCorr=0.0):
        
        
        self.setmean = setmean
                
        self.W = np.load("/home/user/qianyh/dleps_mol_target/checkpoints/dleps_old/denseweight.npy")
        A3, self.con = self._get_con()
        self.full_con = A3.dot(self.W)
        self.genes = pd.read_csv("/home/user/qianyh/dleps_mol_target/checkpoints/dleps_old/corr_genes_12328.csv")
        self.gene_dict=dict(list(zip(self.genes["pr_gene_symbol"], self.genes["pr_gene_id"])))
        self.save_exp = save_exp

        
    
    # The average expression levels for 978 genes across different perturbations in experiments
    def _get_con(self):
        # benchmark = pd.read_csv('/home/user/qianyh/dleps_mol_target/checkpoints/dleps_old/denseweight.npy')
        # A3 = np.concatenate((np.array([1]),benchmark['1.0'].values),axis=0)
        # con = benchmark['1.0'].values
        benchmark = pickle.load(open("/home/user/qianyh/ai_mol_target/data/dleps_final/benchmark.pkl","rb"))
        A3 = np.concatenate((np.array([1]),benchmark['values']),axis=0)
        con = benchmark["values"]
        
        return A3, con
    
    def get_L12k(self,expr):
        # map 12328 gene expressions from 978 landmark genes
        abs_expr = expr + self.con
        A2 = np.hstack([np.ones([expr.shape[0],1]), abs_expr])
        L12k = A2.dot(self.W)
        if self.setmean:
            L12k_df = pd.DataFrame(L12k, columns=self.genes["pr_gene_id"])
            L12k_df = L12k_df - L12k_df.mean()
        else:
            L12k_delta = L12k-self.full_con
            L12k_df = pd.DataFrame(L12k_delta, columns=self.genes["pr_gene_id"])
        if self.save_exp is not None:
            L12k_df.to_csv(self.save_exp)

        return L12k_df