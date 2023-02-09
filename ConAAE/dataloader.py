import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os


class ATAC_Dataset(Dataset):
    def __init__(self, datadir, labeldir, mode='train'):
        self.datadir = datadir
        self.labeldir = labeldir
        self.mode=mode
        self.data,self.labels = self._load_atac_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        atac_sample = self.data[idx]
        cluster = self.labels[idx]
        return {'tensor': torch.from_numpy(atac_sample).float(), 'label': int(cluster)}
        #return {'tensor': torch.from_numpy(atac_sample).float()}

    def _load_atac_data(self):
        data = pd.read_csv(self.datadir, index_col=0)
        #data = data.transpose().values
        data = data.values
        #atac_emb=pd.read_csv(self.datadir,skiprows=1,header=None,index_col=0,sep=' ')
        #atac_emb=atac_emb.values
        
        
        labels = pd.read_csv(self.labeldir, index_col=0)
        labels = labels.values
        labels = np.squeeze(labels)
        #print(labels.shape)  
        #data = labels.merge(data, left_index=True, right_index=True)
        
        #data = data.values
        '''split=atac_emb.shape[0]*0.8
        if self.mode=='train':
          return atac_emb[0:int(split)],labels[0:int(split)]
        elif self.mode == 'test':
          return atac_emb[int(split):],labels[int(split):]
        elif self.mode == 'integration':
          return atac_emb,labels
        else:
          raise KeyError("Mode %s is invalid, must be 'train' or 'test'" % self.mode)'''
          
        split=data.shape[0]*0.8
        if self.mode=='train':
          return data[0:int(split)],labels[0:int(split)]
        elif self.mode == 'test':
          return data[int(split):],labels[int(split):]
        elif self.mode == 'integration':
          return data,labels
        else:
          raise KeyError("Mode %s is invalid, must be 'train' or 'test'" % self.mode)


class RNA_Dataset(Dataset):
    def __init__(self, datadir, labeldir, mode='integration'):
        self.datadir = datadir
        self.labeldir=labeldir
        self.mode=mode
        self.data,self.labels = self._load_rna_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rna_sample = self.data[idx]
        cluster = self.labels[idx]
        '''coro1a = rna_sample[5849]
        rpl10a = rna_sample[2555]'''
        return {'tensor': torch.from_numpy(rna_sample).float(), 'label': int(cluster)}
        #return {'tensor': torch.from_numpy(rna_sample).float()}

    def _load_rna_data(self):
        data = pd.read_csv(self.datadir, index_col=0)
        data = data.values
        #data = data.transpose().values
        #rna_emb=pd.read_csv(self.datadir,skiprows=1,header=None,index_col=0,sep=' ')
        #rna_emb=rna_emb.values
        
        
        labels = pd.read_csv(self.labeldir, index_col=0)
        labels = labels.values
        labels = np.squeeze(labels)
        
        #labels = pd.read_csv(self.labeldir, index_col=0)

        #data = labels.merge(data, left_index=True, right_index=True)
        #data = data.values
        #print(len(data))
        '''split=rna_emb.shape[0]*0.8
        if self.mode=='train':
          return rna_emb[0:int(split)],labels[0:int(split)]
        elif self.mode == 'test':
            return rna_emb[int(split):],labels[int(split):]
        elif self.mode == 'integration':
            return rna_emb, labels
        else:
            raise KeyError("Mode %s is invalid, must be 'train' or 'test'" % self.mode)'''
        
        split=data.shape[0]*0.8
        if self.mode=='train':
          return data[0:int(split)],labels[0:int(split)]
        elif self.mode == 'test':
            return data[int(split):],labels[int(split):]
        elif self.mode == 'integration':
            return data, labels
        else:
            raise KeyError("Mode %s is invalid, must be 'train' or 'test'" % self.mode)


def print_nuclei_names():
    dataset = NucleiDatasetNew(datadir="data/nuclear_crops_all_experiments", mode='test')
    for sample in dataset:
        print(sample['name'])

def test_nuclei_dataset():
    dataset = NucleiDatasetNew(datadir="data/nuclear_crops_all_experiments", mode='train')
    print(len(dataset))
    sample = dataset[0]
    print(sample['image_tensor'].shape)
    print(sample['binary_label'])

    labels = 0
    for sample in dataset:
        print(sample['binary_label'])
        #labels += sample['binary_label']
    #print(labels)

def test_atac_loader():
    dataset = ATAC_Dataset(datadir="mydata")
    #print(len(dataset))
    sample = dataset[0]
    print(torch.max(sample['tensor']))
    print(sample['tensor'].shape)
    for k in sample.keys():
        print(k)
        print(sample[k])

def test_rna_loader():
    dataset = RNA_Dataset(datadir="data/nCD4_gene_exp_matrices")
    print(len(dataset))
    sample = dataset[0]
    print(torch.max(sample['tensor']))
    print(sample['tensor'].shape)
    for k in sample.keys():
        print(k)
        print(sample[k])
        
def test_more_atac_loader():
    dataset=ATAC_Dataset(datadir="mydata")
    #print(len(dataset))
    #print(type(dataset))
    sample=[]
    for idx,i in enumerate(dataset):
      if(idx==10): break
      sample.append(i)
    for i in sample:
      print(i['binary_label'])

def test_more_rna_loader():
    dataset=RNA_Dataset(datadir="data/nCD4_gene_exp_matrices")
    sample=[]
    for idx,i in enumerate(dataset):
      if(idx==10): break
      sample.append(i)
    for i in sample:
      print(i['binary_label'])


if __name__ == "__main__":
    test_nuclei_dataset()
