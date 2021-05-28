import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from skimage import io

import os

class ToTensorNormalize(object):
    """Convert ndarrays in sample to Tensors."""
 
    def __call__(self, sample):
        image_tensor = sample['image_tensor']
 
        # rescale by maximum and minimum of the image tensor
        minX = image_tensor.min()
        maxX = image_tensor.max()
        image_tensor=(image_tensor-minX)/(maxX-minX)
 
        # resize the inputs
        # torch image tensor expected for 3D operations is (N, C, D, H, W)
        image_tensor = image_tensor.max(axis=0)
        image_tensor = cv2.resize(image_tensor, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        image_tensor = np.clip(image_tensor, 0, 1)
        return torch.from_numpy(image_tensor).view(1, 64, 64)

class NucleiDatasetNew(Dataset):
    def __init__(self, datadir, mode='train', transform=ToTensorNormalize()):
        self.datadir = datadir
        self.mode = mode
        self.images = self.load_images()
        self.transform = transform
        self.threshold = 0.74

    # Utility function to load images from a HDF5 file
    def load_images(self):
        # load labels
        label_data = pd.read_csv(os.path.join(self.datadir, "ratio.csv"))
        label_data_2 = pd.read_csv(os.path.join(self.datadir, "protein_ratios_full.csv"))
        label_data = label_data.merge(label_data_2, how='inner', on='Label')
        label_dict = {name: (float(ratio), np.abs(int(cl)-2)) for (name, ratio, cl) in zip(list(label_data['Label']), list(label_data['Cor/RPL']), list(label_data['mycl']))}
        label_dict_2 = {name: np.abs(int(cl)-2) for (name, cl) in zip(list(label_data_2['Label']), list(label_data_2['mycl']))}
        del label_data
        del label_data_2

        # load images
        images_train = []
        images_test = []

        for f in os.listdir(os.path.join(self.datadir, "images")):
            basename = os.path.splitext(f)[0]
            fname = os.path.join(os.path.join(self.datadir, "images"), f)
            if basename in label_dict.keys():
                images_test.append({'name': basename, 'label': label_dict[basename][0], 'image_tensor': np.float32(io.imread(fname)), 'binary_label': label_dict[basename][1]})
            else:
                try:
                    images_train.append({'name': basename, 'label': -1, 'image_tensor': np.float32(io.imread(fname)), 'binary_label': label_dict_2[basename]})
                except Exception as e:
                    pass

        if self.mode == 'train':
            return images_train
        elif self.mode == 'test':
            return images_test
        else:
            raise KeyError("Mode %s is invalid, must be 'train' or 'test'" % self.mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]

        if self.transform:
            # transform the tensor and the particular z-slice
            image_tensor = self.transform(sample)
            return {'image_tensor': image_tensor, 'name': sample['name'], 'label': sample['label'], 'binary_label': sample['binary_label']}
        return sample

class ATAC_Dataset(Dataset):
    def __init__(self, datadir, mode='train'):
        self.datadir = datadir
        self.mode=mode
        self.atac_data,self.labels = self._load_atac_data()

    def __len__(self):
        return len(self.atac_data)

    def __getitem__(self, idx):
        atac_sample = self.atac_data[idx]
        cluster = self.labels[idx]
        return {'tensor': torch.from_numpy(atac_sample).float(), 'binary_label': int(cluster)}
        #return {'tensor': torch.from_numpy(atac_sample).float()}

    def _load_atac_data(self):
        data = pd.read_csv(os.path.join(self.datadir, "ATAC_seq.csv"), index_col=0)
        #print(data.shape)
        data = data.transpose()
        labels = pd.read_csv(os.path.join(self.datadir, "label.csv"), index_col=0)
        #print(labels.shape)  
        data = labels.merge(data, left_index=True, right_index=True)
        data = data.values
        #print(len(data))
        #train,train_labels=data[0:1433,1:],data[0:1433,0]
        #print(len(train))
        #print(len(train_labels))
        #test,test_labels=data[1433:,1:],data[1433:,0]
        
        if self.mode=='train':
          return data[0:1433,1:],data[0:1433,0]
        elif self.mode == 'test':
            return data[1433:,1:],data[1433:,0]
        else:
            raise KeyError("Mode %s is invalid, must be 'train' or 'test'" % self.mode)


class RNA_Dataset(Dataset):
    def __init__(self, datadir,mode='train'):
        self.datadir = datadir
        self.mode=mode
        self.rna_data,self.labels = self._load_rna_data()

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, idx):
        rna_sample = self.rna_data[idx]
        cluster = self.labels[idx]
        '''coro1a = rna_sample[5849]
        rpl10a = rna_sample[2555]'''
        return {'tensor': torch.from_numpy(rna_sample).float(), 'binary_label': int(cluster)}
        #return {'tensor': torch.from_numpy(rna_sample).float()}

    def _load_rna_data(self):
        data = pd.read_csv(os.path.join(self.datadir, "RNA_seq.csv"), index_col=0)
        data = data.transpose()
        labels = pd.read_csv(os.path.join(self.datadir, "label.csv"), index_col=0)

        data = labels.merge(data, left_index=True, right_index=True)
        data = data.values
        
        if self.mode=='train':
          return data[0:1433,1:],data[0:1433,0]
        elif self.mode == 'test':
            return data[1433:,1:],data[1433:,0]
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
