# MoaNet
source code for MoaNet: Learning protein-ligand mechanism of action from transcriptional profiles and sequence information

## Installation

```bash
conda create -n moanet python=3.10 
conda activate moanet

# install required python dependencies
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 numpy==1.26.4 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch_geometric==2.5.3
pip install tensorflow==2.14.0
pip install fair-esm==1.0.2
```


## Usage

### 1.Predicting agonists/antagonists for specific target(repurposing or virtual screening)

step1: prepare molecular embedding for molecular sets by kpgt(please refering to https://github.com/lihan97/KPGT.git and our demo:/home/qianyh/MoANet/get_kpgt_latent_for_smiles.ipynb to get d_graph_dict.pkl)

step2: run run_purposing.ipynb to predict agonists/antagonists for virtual screening.

### 2. training model for yourself

#### Benchmark dataset 

The benchmark datasets for training and inference are available in data/moa_dataset and we conduct a scaffold-based spliting strategy and protein-based spliting strategy for testing the generalization performance of unseen drugs and unseen proteins 

#### Model training and other baseline model

Please refer to train_moanet.ipynb for model training. Baseline models can also be reproduced by modifying the training configuration.

#### drugsig encoder implementation

There are option module by merging drug-induced gene signature by "drugsig" module. We find that module can improves generalization performance on unseen proteins.

The training code for drugsig encoder is available in training_drugsig_encoder.ipynb
The pretrained model weights are available in ckpts/dleps_fold_0_seed_0.ckpt











