import os           
import copy         
from time import time
from prettytable import PrettyTable
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score,f1_score

from torch_geometric.data import Data,Batch
import random
import numpy as np             
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import MoaNet

from dti_dataset import DTI_Dataset


class Trainer:
    
    def __init__(self,model,**config):
        super(Trainer,self).__init__()
        
        self.config = config
        self.result_folder = config["result_folder"]
        
        if 'num_workers' not in self.config.keys():
            self.config['num_workers'] = 6
        if 'decay' not in self.config.keys():
            self.config['decay'] = 0
            
        self._set_seed(config["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
    def _set_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def cal_score(self,feat):
        if self.config["use_drugsig"]:
            if self.config["protein_model_type"] == "supervised":
                d_s = feat[0].to(self.device)
                d_expr = feat[1].to(self.device)
                p_s = feat[2].to(self.device)
                mask = feat[3].to(self.device)
                p_expr = feat[4].to(torch.float32).to(self.device)
                label = feat[5].to(self.device)
                score = self.model((d_s,d_expr,p_s,mask,p_expr))     
                
            elif self.config["protein_model_type"] in ["esm2","prott5","protein_cnn"]:
                d_s = feat[0].to(self.device)
                d_expr = feat[1].to(self.device)
                p_s = feat[2].to(self.device)
                p_expr = feat[3].to(torch.float32).to(self.device)
                label = feat[4].to(self.device)
                score = self.model((d_s,d_expr,p_s,p_expr))                               
        else:
            if self.config["protein_model_type"] == "supervised":
                d_s = feat[0].to(self.device)
                p_s = feat[1].to(self.device)
                mask = feat[2].to(self.device)
                p_expr = feat[3].to(torch.float32).to(self.device)
                label = feat[4].to(self.device)
                score = self.model((d_s,p_s,mask,p_expr))     
                
            elif self.config["protein_model_type"] in ["esm2","prott5"]:
                d_s = feat[0].to(self.device)
                p_s = feat[1].to(self.device)
                p_expr = feat[2].to(torch.float32).to(self.device)
                label = feat[3].to(self.device)
                score = self.model((d_s,p_s,p_expr))     
            
            elif self.config["protein_model_type"] =="protein_cnn":
                d_s = feat[0].to(self.device)
                p_s = feat[1].to(self.device)
                p_expr = feat[2].to(torch.float32).to(self.device)
                label = feat[3].to(self.device)
                score = self.model((d_s,p_s,p_expr))           
    
        return score,label 
    
    def train(self,train,val=None,test=None,verbose=True):
        
        lr = self.config["LR"]
        decay = self.config["decay"]
        batch_size = self.config["batch_size"]
        train_epoch = self.config['train_epoch']
        early_stop_threshold = self.config["early_stop"]
        
        opt = torch.optim.Adam(self.model.parameters(),lr=lr,weight_decay=decay)
        
        if verbose:
            print('--- Data Preparation ---')
        
        training_dataset = DTI_Dataset(train,**self.config)
        training_generator = DataLoader(training_dataset,batch_size=batch_size,shuffle=True,collate_fn=training_dataset.collate_fn,drop_last=True)
        
        if val is not None:
            validation_dataset = DTI_Dataset(val,**self.config)
            validation_generator = DataLoader(validation_dataset,batch_size=batch_size,shuffle=False,collate_fn=validation_dataset.collate_fn,drop_last=True)

        if test is not None:
            testing_dataset = DTI_Dataset(data_df=test,**self.config)
            testing_generator = DataLoader(testing_dataset,batch_size=batch_size,shuffle=False,collate_fn=testing_dataset.collate_fn,drop_last=True)
        
        max_auc = 0
        self.model_max = copy.deepcopy(self.model)
        
        float2str = lambda x:'%0.4f'%x
        if verbose:
            print('--- Go for Training ---')
            
        t_start = time() 
        iteration_loss = 0
        
        for epo in range(train_epoch):
            
            #val phase
            self.model.eval()
            if val is not None:
                with torch.no_grad():
                    auc, auprc, f1, sensitivity, specificity, accuracy = self.value(validation_generator, self.model)
                    #lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, f1]))
                    #valid_metric_record.append(lst)
                    
                    if verbose:
                        print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
                    ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , sensitivity: ' + str(sensitivity)[:7] + " , specificity: " + str(specificity)[:7] + " , accuracy: " + str(accuracy)[:7])
                                       
                    if auc > max_auc:
                        self.model_max = copy.deepcopy(self.model)
                        max_auc = auc
                        es = 0
                    else:
                        es += 1
                        print(f"Counter {es} of {early_stop_threshold}")
                        if es > early_stop_threshold-1:
                            print("Early stopping with best_auc: ", str(max_auc)[:7], "and auc for this epoch: ", str(auc)[:7], "...")
                            break
            
            else:
                self.model_max = copy.deepcopy(self.model)
        
            self.model.train()
            #training phase
            for i, feat in enumerate(training_generator):
                score,label = self.cal_score(feat)
                loss_fct = torch.nn.BCELoss()
                m = torch.nn.Sigmoid()
                n = torch.squeeze(m(score), 1)
                loss = loss_fct(n,label)
                
                #loss_history.append(loss.item())
                iteration_loss += 1
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if verbose:
                    if (i % 500 == 0):
                        t_now = time()
                        print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
                            ' with loss ' + str(loss.cpu().detach().numpy())[:7] +\
                            ". Total time " + str(int(t_now - t_start))[:7] + " seconds") 
            
        if test is not None:
            if verbose:
                print('--- Go for Testing ---')
                
            auc, auprc, f1, sensitivity, specificity, accuracy= self.value(testing_generator, self.model_max)
            # test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
            # test_table.add_row(list(map(float2str, [auc, auprc, f1])))
            if verbose:
                print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
                    ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , sensitivity: ' + str(sensitivity)[:7] + " , specificity: " + str(specificity)[:7] + " , accuracy: " + str(accuracy)[:7])               
            
            torch.save(self.model.state_dict(),os.path.join(self.result_folder,"model.pt"))
            
    def value(self,data_generator,model,repurposing_mode=False):
        
        y_pred = []
        y_label = []
        model.eval()
        
        for i,feat in enumerate(data_generator):
            score,label = self.cal_score(feat)
            
            m = torch.nn.Sigmoid()
            logits = torch.squeeze(m(score)).detach().cpu().numpy()
            
            label_ids = Variable(label)
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
            
        model.train()
        
        if repurposing_mode:
            return y_pred,y_label
        
        else:
            import joblib
            joblib.dump(y_label,os.path.join(self.result_folder,"ylabel.pkl"))
            joblib.dump(y_pred,os.path.join(self.result_folder,"ypred.pkl"))
            
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
           
            precision1 = precision_score(y_label, y_pred_s)
            
            return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), np.max(f1[5:]), sensitivity, specificity, accuracy
        