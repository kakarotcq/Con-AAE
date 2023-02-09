import torch
import torch.utils.data
from dataloader import RNA_Dataset
from dataloader import ATAC_Dataset
from model import FC_VAE,FC_Classifier,FC_Autoencoder,Simple_Classifier
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

genomics_dataset = RNA_Dataset(datadir="./data/10X_PBMC/rna.csv",labeldir="./data/10X_PBMC/label.csv",mode='test')
ATAC_dataset = ATAC_Dataset(datadir="./data/10X_PBMC/atac.csv",labeldir="./data/10X_PBMC/label.csv",mode='test')


#genomics_dataset = RNA_Dataset(datadir="./data/SHARE/rna.csv",labeldir="./data/SHARE/label.csv",mode='test')
#ATAC_dataset = ATAC_Dataset(datadir="./data/SHARE/atac.csv",labeldir="./data/SHARE/label.csv",mode='test')

RNA_loader=torch.utils.data.DataLoader(genomics_dataset,batch_size=100,drop_last=False,shuffle=False)
ATAC_loader=torch.utils.data.DataLoader(ATAC_dataset,batch_size=100,drop_last=False,shuffle=False)

ATAC_dim=ATAC_dataset.atac_data.shape[1]
genomics_dim=genomics_dataset.rna_data.shape[1]
label_num=len(np.unique(ATAC_dataset.labels))

cell_num=ATAC_dataset.atac_data.shape[0]

netRNA = FC_Autoencoder(n_input=genomics_dim, nz=50,n_hidden=genomics_dim)
netATAC = FC_Autoencoder(n_input=ATAC_dim, nz=50,n_hidden=ATAC_dim)


netRNA.cuda()
netATAC.cuda()

#netRNA = FC_Autoencoder(n_input=2613, nz=50,n_hidden=2613)
#netATAC = FC_Autoencoder(n_input=815, nz=50,n_hidden=815)

#netRNA = FC_VAE(n_input=2613, nz=50,n_hidden=2613)

#netATAC = FC_VAE(n_input=815, nz=50,n_hidden=815)

def mink(arraylist,k):
    minlist=[]
    minlist_id=list(range(0,k))
    m=[minlist,minlist_id]
    for i in minlist_id:
        minlist.append(arraylist[i])
    for i in range(k,len(arraylist)):
        if arraylist[i]<max(minlist):
            mm=minlist.index(max(minlist))
            del m[0][mm]
            del m[1][mm]
            m[0].append(arraylist[i])
            m[1].append(i)
    return minlist_id
#netClf=FC_Classifier(nz=50,n_out=3)
netConClf=Simple_Classifier(nz=50)
netConClf.cuda()

netRNA.eval()
netATAC.eval()
netConClf.eval()

#netRNA.cuda()
#netATAC.cuda()
#netClf.cuda()


def find_mutual_nn(data1, data2, k1, k2, n_jobs):
     k_index_1 = cKDTree(data1).query(x=data2, k=k1, workers=n_jobs)[1]
     k_index_2 = cKDTree(data2).query(x=data1, k=k2, workers=n_jobs)[1]
     #print(k_index_1)
     #print(k_index_2)
     mutual_1 = []
     mutual_2 = []
     sum=0
     for index_2 in range(data2.shape[0]):
         for index_1 in k_index_1[index_2]:
             if index_2 in k_index_2[index_1]:
                 mutual_1.append(index_1)
                 mutual_2.append(index_2)
                 if index_1==index_2:
                   sum+=1
     return sum


def eva():

  #netConClf.load_state_dict(torch.load("AE_condi_adv/netCondClf_1000.pth"))
  
  
  predicted_label=np.zeros(shape=(0,),dtype=int)
  ATAC_class_label=np.zeros(shape=(0,),dtype=int)
  RNA_latent=np.zeros(shape=(0,50))
  ATAC_latent=np.zeros(shape=(0,50))
  RNA_recon=np.zeros(shape=(0,genomics_dim))
  ATAC_recon=np.zeros(shape=(0,ATAC_dim))
  
  for idx, (rna_samples,atac_samples) in enumerate(zip(RNA_loader,ATAC_loader)):
    rna_inputs=rna_samples['tensor']
    rna_labels=rna_samples['binary_label']
    atac_inputs=atac_samples['tensor']
    atac_labels=atac_samples['binary_label'] 
    
    #print(type(rna_inputs))
    #input('pause')
    #print(ATAC_label.shape)
    #print(atac_labels.shape)
    #print(atac_labels.numpy().shape)
    ATAC_class_label=np.concatenate((ATAC_class_label,atac_labels.numpy()),axis=0)
    
    #rna_inputs=rna_inputs.cuda()
    #atac_inputs=atac_inputs.cuda()
    
    rna_latents,_=netRNA(rna_inputs.cuda())
    atac_latents,_=netATAC(atac_inputs.cuda())
    
    #_,rna_latents,_,_=netRNA(rna_inputs)
    #_,atac_latents,_,_=netATAC(atac_inputs)
    
    atac_recon=netATAC.decoder(atac_latents)
    rna_recon=netRNA.decoder(rna_latents)
    
    #print(netConClf(atac_latents))
    
    predicted_label=np.concatenate((predicted_label,np.argmax(netConClf(atac_latents).cpu().detach().numpy(),axis=1)))
    #print(RNA_predicted_label)
    #break
    
    RNA_latent=np.concatenate((RNA_latent,rna_latents.cpu().detach().numpy()),axis=0)
    ATAC_latent=np.concatenate((ATAC_latent,atac_latents.cpu().detach().numpy()),axis=0)
    
    RNA_recon=np.concatenate((RNA_recon,rna_recon.cpu().detach().numpy()),axis=0)
    ATAC_recon=np.concatenate((ATAC_recon,atac_recon.cpu().detach().cpu().numpy()),axis=0)
  
  
  
  #print(ATAC_label.shape)
  pd.DataFrame(RNA_latent).to_csv('embedding/modal_anchor_test/pbmc_rna_emb.csv')
  pd.DataFrame(ATAC_latent).to_csv('embedding/modal_anchor_test/pbmc_atac_emb.csv')
  
  #pd.DataFrame(RNA_recon).to_csv('embedding/conAAE_SHARE_rna_recon.csv')
  #pd.DataFrame(ATAC_recon).to_csv('embedding/conAAE_SHARE_atac_recon.csv')
  #input('pause')
  #for k in range(1,6):
  #k=k*10
  #print("When k = %d"%k)
  pairlist=[]
  classlist=[]
  distlist=[]
  flag=np.zeros(cell_num)
  kright10=0
  kright20=0
  kright30=0
  kright40=0
  kright50=0
    
  for index,i in enumerate(ATAC_latent):
    mini=999999
    minindex=0
    distlist.clear()
    for idx,j in enumerate(RNA_latent):
      dist=np.linalg.norm(i-j)
      distlist.append(dist)
        #print(dist)
      if(dist<mini and flag[idx]==0):
        mini=dist
        minindex=idx
    kindex10=mink(distlist,10)
    kindex20=mink(distlist,20)
    kindex30=mink(distlist,30)
    kindex40=mink(distlist,40)
    kindex50=mink(distlist,50)
    if(index in kindex10):
      kright10+=1
    if(index in kindex20):
      kright20+=1
    if(index in kindex30):
      kright30+=1
    if(index in kindex40):
      kright40+=1
    if(index in kindex50):
      kright50+=1
    flag[minindex]=1
    pairlist.append(minindex)
    classlist.append(ATAC_class_label[minindex])
    
  ATAC_seq=np.arange(0,cell_num,1)
    
    #print(len(pairlist))
  pairlist=np.array(pairlist)
    
    #print(flag)
    
  classlist=np.array(classlist)
    #print(pairlist)
    #print(classlist)
  print(float(kright10)/cell_num)
  print(float(kright20)/cell_num)
  print(float(kright30)/cell_num)
  print(float(kright40)/cell_num)
  print(float(kright50)/cell_num)
  print(float(np.sum(pairlist==ATAC_seq)))
  print(float(np.sum(ATAC_class_label==classlist)/cell_num))
  pd.DataFrame(classlist).to_csv('./embedding/modal_anchor_test/pbmc_label_transferred.csv')
  #input('pause')
  #print(type(ATAC_label))
  #print(pairlist.shape)
  #print(RNA_latent.shape)
  #print(ATAC_latent.shape)
  #print(RNA_recon.shape)
  #print(ATAC_recon.shape)
  
  sum=find_mutual_nn(RNA_latent,ATAC_latent,10,10,1)
  print(sum)
  

for i in range(1,2):
  print("when epoch = %d"%(i*100))
  #netRNA.load_state_dict(torch.load("conAAE_PBMC_1000/netRNA_DE_"+str(100*i)+".pth"))
  #netATAC.load_state_dict(torch.load("conAAE_PBMC_1000/netATAC_DE_"+str(100*i)+".pth"))
  
  netRNA.load_state_dict(torch.load("modal_anchor_PBMC/netRNA_DE_"+str(100)+".pth"))
  netATAC.load_state_dict(torch.load("modal_anchor_PBMC/netATAC_DE_"+str(100)+".pth"))
  
  #input('pause')
  eva()
  #input('pause')