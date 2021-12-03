import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import seaborn as sns
from sklearn.decomposition import PCA

ATAC=pd.read_csv("sim_ATAC_6000_s5.csv",index_col=0)
labels=pd.read_csv("sim_label_6000.csv",index_col=0)
RNA=pd.read_csv("sim_RNA_6000_s5.csv",index_col=0)

ATAC=ATAC.transpose()
ATACX=ATAC.values

RNA=RNA.transpose()
RNAX=RNA.values


#print(ATACX.shape)
#print(RNAX.shape)

#input()
ATAC_obs=labels.merge(ATAC,left_index=True,right_index=True)
#for i in range(ATACX.shape[1]):
#    ATAC_obs=ATAC_obs.drop([i],axis=1)
#ATACX=ATACX[:,0:RNAX.shape[1]]
ATAC_obs=ATAC_obs.drop(columns=ATAC.columns)
#print(ATAC_obs)
#input()
#pca=PCA(n_components=500)

#ATACX=pca.fit_transform(ATACX)

#print(ATACX)
#print(ATACX.shape)

#input()
ATAC_var=[i for i in range(ATACX.shape[1])]
ATAC_var=pd.DataFrame(index=ATAC_var)
ATAC_adata=ad.AnnData(ATACX,obs=ATAC_obs,var=ATAC_var)

#print(ATAC_adata)



RNA_obs=labels.merge(RNA,left_index=True,right_index=True)
#for i in range(RNAX.shape[1]):
#    RNA_obs=RNA_obs.drop([i],axis=1)

RNA_obs=RNA_obs.drop(columns=RNA.columns)
#print(RNA_obs)

pca=PCA(n_components=ATACX.shape[1])

RNAX=pca.fit_transform(RNAX)

print(RNAX.shape)

RNA_var=[i for i in range(RNAX.shape[1])]
RNA_var=pd.DataFrame(index=RNA_var)
RNA_adata=ad.AnnData(RNAX,obs=RNA_obs,var=RNA_var)

print(RNA_adata)
#input()
#ingest

#var_names=ATAC_adata.var_names.intersection(RNA_adata.var_names)

#ATAC_adata=ATAC_adata[:,var_names]

#RNA_adata=RNA_adata[:,var_names]

sc.pp.pca(RNA_adata)
sc.pp.neighbors(RNA_adata)
sc.tl.umap(RNA_adata)
#sc.pl.umap(adata_ref,)

print(RNA_adata)
#input()
print(ATAC_adata)
#input()
sc.tl.ingest(ATAC_adata,RNA_adata,obs='sim-labels')
#input()
count=0
for i in range(ATAC_adata.X.shape[0]):
    if(labels['sim-labels'][i]==ATAC_adata.obs['sim-labels'][i]):
        count+=1

print(float(count)/float(ATAC_adata.X.shape[0]))
