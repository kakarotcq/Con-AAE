# Con-AAE: Contrastive Cycle Adversarial Autoencoders for Single-cell Multi-omics Alignment and Integration
This model can integrate and align multi-omics data.

## Setup
Con-AAE is a deep learning framework based on Pytorch. All experiments of Con-AAE are implemented with **python version=3.7.7** and **pytorch version=1.4.0**.
We exported the environment, so users can create environment with:
```
conda env create -f conAAE_env.yaml
```

## Usage
Required Files:
1. gene count matrix file (rna.csv): rows are cells and columns are genes.
2. scATAC matrix file (atac.csv): rows care cells and columns are peaks.
3. annotations for cells (label.csv): used to annotate the cell type for each cell.

Parameters:
```
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir")
    options.add_argument('-i', '--input-dir', action="store", dest="input_dir")
    options.add_argument('--save-freq', action="store", dest="save_freq", default=10, type=int)
    options.add_argument('--pretrained-file', action="store")

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=32, type=int) #batch size in training process
    options.add_argument('-nz', '--latent-dimension', action="store", dest="nz", default=50, type=int)  #dimension of embeddings
    
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-4, type=float) # learning rate for discriminator network
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=101, type=int) # max epoch
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float) # weight decay in training process
    options.add_argument('--contrastive-loss',action="store_true") # whether calculate contrastive loss when training the model
    options.add_argument('--consistency-loss',action="store_true") # whether calculate cycle consistency loss when training the model
    options.add_argument('--anchor-loss',action="store_true") # whether introducing pairwise information in training process
    options.add_argument('--MMD-loss',action="store_true") # whether introducing mmd loss in embedding space
    options.add_argument('--augmentation',action="store_true") # whether implementing augmentation for data

    # hyperparameters
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-4, type=float) #learning rate for coupled autoencoders
    options.add_argument('--margin', action="store", default=0.3, type=float) # margin in contrastive loss
    options.add_argument('--alpha', action="store", default=10.0, type=float) # weight of contrastive loss, discriminative loss and cycle consistency loss 
    options.add_argument('--beta', action="store", default=1., type=float) # weight of simple classifier
    options.add_argument('--beta1', action="store", default=0.5, type=float) # parameter in Adam optimizer
    options.add_argument('--beta2', action="store", default=0.999, type=float) # parameter in Adam optimizer
```
## Example
An example is placed as **Demo.ipynb**. Please run the example to get familiar with Con-AAE.
It may take 10 mins with one NVIDIA tesla v100 GPU or 50 mins with only CPU.
