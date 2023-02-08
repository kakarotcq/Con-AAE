# Con-AAE: Contrastive Cycle Adversarial Autoencoders for Single-cell Multi-omics Alignment and Integration
This model can integrate and align multi-omics data.

## Setup
Con-AAE is a deep learning framework based on Pytorch. All expereiments of Con-AAE are implemented with **python version=3.7.7** and **pytorch version=1.4.0**
We exported the environment by conda, so users can create environment with:
```
conda env create -f conAAE_env.yaml
```

## Usage
Required Files:
1.gene count matrix file (rna.csv): rows are cells and columns are genes.
2.scATAC matrix file (atac.csv): rows care cells are peaks.
3.annotations for cells (label.csv): used to annotate the cell type for each cell.

## Example
An example is placed as **Demo.ipynb**. Please run the example to get familiar with Con-AAE.
It may take 10 mins with one NVIDIA tesla v100 GPU or 50 mins with only CPU.
