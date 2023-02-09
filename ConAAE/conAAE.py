import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import pandas as pd

#from dataloader import RNA_Dataset
#from dataloader import ATAC_Dataset
from ConAAE.model import FC_Autoencoder, FC_Classifier, FC_VAE, Simple_Classifier,TripletLoss

import os
import argparse

import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.manual_seed(1)
def accuracy(output, target):
    pred = output.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).float().sum().item()
    return correct

def dis_accuracy(output,target):
    zero=torch.zeros_like(output)
    one=torch.ones_like(output)
    output=torch.where(output<0.5,zero,output)
    output=torch.where(output>0.5,one,output)
    accuracy=(output==target).sum().item()
    return accuracy
    
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

class conAAE:
  def __init__(self,data1,data2,model_parameters):
    self.RNA=data1
    self.ATAC=data2
    self.args=model_parameters
    self.ATAC_loader = torch.utils.data.DataLoader(self.ATAC, batch_size=self.args.batch_size, drop_last=False, shuffle=False)
    self.RNA_loader = torch.utils.data.DataLoader(self.RNA, batch_size=self.args.batch_size, drop_last=False, shuffle=False)
    self.label_num=len(np.unique(self.RNA.labels))
    self.cell_num=self.RNA.data.shape[0]
    
    
    self.netRNA=FC_Autoencoder(n_input=self.RNA.data.shape[1],n_hidden=self.RNA.data.shape[1],nz=self.args.nz)
    self.netATAC=FC_Autoencoder(n_input=self.ATAC.data.shape[1],n_hidden=self.ATAC.data.shape[1],nz=self.args.nz)
    
    self.netCondClf = Simple_Classifier(nz=self.args.nz,n_out=self.label_num)
    self.netClf=FC_Classifier(nz=self.args.nz,n_out=1)
    
    if self.args.use_gpu:
      self.netRNA.cuda()
      self.netATAC.cuda()
      self.netCondClf.cuda()
      self.netClf.cuda()
    self.opt_netRNA = optim.Adam(list(self.netRNA.parameters()), lr=self.args.learning_rate_AE,betas=(self.args.beta1,self.args.beta2),weight_decay=0.0001)
    self.opt_netClf = optim.Adam(list(self.netClf.parameters()), lr=self.args.learning_rate_D, weight_decay=self.args.weight_decay)
    self.opt_netATAC = optim.Adam(list(self.netATAC.parameters()), lr=self.args.learning_rate_AE,betas=(self.args.beta1,self.args.beta2),weight_decay=0.0001)
    self.opt_netCondClf = optim.Adam(list(self.netCondClf.parameters()), lr=self.args.learning_rate_AE)
    self.criterion_reconstruct = nn.MSELoss()
    self.criterion_dis=nn.MSELoss()
    #self.anchor_loss=nn.MSELoss()
    self.criterion_classify = nn.CrossEntropyLoss()
    if self.args.contrastive_loss:
      self.triplet_loss=TripletLoss(self.args.margin)
    if self.args.anchor_loss:
      self.anchor_loss=nn.MSELoss()
      
      
      
  def guassian_kernel(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

  def mmd_rbf(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss
    
    
  def train_autoencoders(self,rna_inputs, atac_inputs, rna_class_labels=None, atac_class_labels=None):
   
    self.netRNA.train()  
    self.netATAC.train()

    #if args.discriminator:
    self.netClf.eval()
    #if args.conditional:
    self.netCondClf.train()
    
    # process input data
    rna_inputs, atac_inputs = Variable(rna_inputs), Variable(atac_inputs)

    if self.args.use_gpu:
        rna_inputs, atac_inputs = rna_inputs.cuda(), atac_inputs.cuda()

    # reset parameter gradients
    self.netRNA.zero_grad()
    self.netATAC.zero_grad()
    self.netCondClf.zero_grad()

    # forward pass
    rna_latents,rna_recon=self.netRNA(rna_inputs)
    atac_latents,atac_recon=self.netATAC(atac_inputs)

    rna_scores = self.netClf(rna_latents)
    atac_scores = self.netClf(atac_latents)
    rna_scores=torch.squeeze(rna_scores,dim=1)
    atac_scores=torch.squeeze(atac_scores,dim=1)
    rna_labels = torch.zeros(rna_scores.size(0),).float()
    atac_labels = torch.ones(atac_scores.size(0),).float()
    
    rna_class_scores = self.netCondClf(rna_latents)
    atac_class_scores = self.netCondClf(atac_latents)
    
    if self.args.use_gpu:
      rna_labels, atac_labels = rna_labels.cuda(), atac_labels.cuda() 
      rna_class_labels, atac_class_labels = rna_class_labels.cuda(), atac_class_labels.cuda()

    rna_recon_loss = self.criterion_reconstruct(rna_inputs, rna_recon)
    atac_recon_loss = self.criterion_reconstruct(atac_inputs, atac_recon)
    # autoencoder training loss
    #loss = self.args.alpha*(rna_recon_loss + atac_recon_loss)
    loss = self.args.alpha*(rna_recon_loss + atac_recon_loss)
    #discriminative loss
    clf_loss = 0.5*self.criterion_dis(rna_scores, atac_labels) + 0.5*self.criterion_dis(atac_scores, rna_labels)
    loss+=self.args.alpha*clf_loss
    
    #simple classifier loss
    clf_class_loss = 0.5*self.criterion_classify(rna_class_scores, rna_class_labels) + 0.5*self.criterion_classify(atac_class_scores, atac_class_labels)
    loss += self.args.beta*clf_class_loss
    
    #contrastive loss
    if self.args.contrastive_loss:
      inputs=torch.cat((atac_latents,rna_latents),0)
      labels=torch.cat((atac_class_labels,rna_class_labels),0)
      tri_loss=self.triplet_loss(inputs,labels)
      loss+=self.args.alpha*tri_loss
    #cycle consistency loss    
    if self.args.consistency_loss:
      atac_latents_recon=self.netRNA.encoder(self.netRNA.decoder(atac_latents))
      rna_latents_recon=self.netATAC.encoder(self.netATAC.decoder(rna_latents))
      atac_latents_recon_loss=self.criterion_reconstruct(atac_latents,atac_latents_recon)
      rna_latents_recon_loss=self.criterion_reconstruct(rna_latents,rna_latents_recon)
      loss+=self.args.alpha*(atac_latents_recon_loss+rna_latents_recon_loss)
    #mmd loss
    if self.args.MMD_loss:
      MMD_loss=mmd_rbf(atac_latents,rna_latents)
      loss+=self.args.alpha*MMD_loss
    if self.args.anchor_loss:
      anchor_loss=self.criterion_reconstruct(rna_latents,atac_latents)
      loss+=0.1*anchor_loss
    # backpropagate and update model
    
    
    
    #print(loss.shape)
    #print(loss)
    
    
    loss.backward()
    self.opt_netRNA.step()
    self.opt_netATAC.step()
    self.opt_netCondClf.step()
    #summary_stats=0
    summary_stats = {'rna_recon_loss': rna_recon_loss.item()*rna_latents.size(0), 'atac_recon_loss': atac_recon_loss.item()*atac_latents.size(0)}
    summary_stats['clf_class_loss'] = clf_class_loss.item()*(rna_latents.size(0)+atac_latents.size(0))
    if self.args.contrastive_loss:
        summary_stats['triplet_loss']=tri_loss*rna_latents.size(0)
    if self.args.anchor_loss:
        summary_stats['anchor_loss']=anchor_loss*rna_latents.size(0)
    if self.args.consistency_loss:
        summary_stats['atac_latents_recon_loss']=atac_latents_recon_loss*atac_latents.size(0)
        summary_stats['rna_latents_recon_loss']=rna_latents_recon_loss*rna_latents.size(0)
    if self.args.MMD_loss:
        summary_stats['MMD_loss']=MMD_loss*rna_latents.size(0)
    summary_stats['clf_loss']=clf_loss.item()*(rna_latents.size(0)+atac_latents.size(0))

    return summary_stats
    
    
  def train_classifier(self,rna_inputs, atac_inputs, rna_class_labels=None, atac_class_labels=None):
     
    self.netRNA.eval()
    self.netATAC.eval()
    self.netClf.train()

    # process input data
    rna_inputs, atac_inputs = Variable(rna_inputs), Variable(atac_inputs)

    if self.args.use_gpu:
        rna_inputs, atac_inputs = rna_inputs.cuda(), atac_inputs.cuda()
    
    # reset parameter gradients
    self.netClf.zero_grad()

    rna_latents, _ = self.netRNA(rna_inputs)
    atac_latents, _ = self.netATAC(atac_inputs)

    rna_scores = self.netClf(rna_latents)
    atac_scores = self.netClf(atac_latents)

    rna_labels = torch.zeros(rna_scores.size(0),).float()
    atac_labels = torch.ones(atac_scores.size(0),).float()
   
    if self.args.use_gpu:
        rna_labels, atac_labels = rna_labels.cuda(), atac_labels.cuda()
        
    rna_scores=torch.squeeze(rna_scores,dim=1)
    atac_scores=torch.squeeze(atac_scores,dim=1)

    # compute losses
    #print(rna_scores.shape)
    #print(rna_labels.shape)
    clf_loss = 0.5*self.criterion_dis(rna_scores, rna_labels) + 0.5*self.criterion_dis(atac_scores, atac_labels)

    loss = clf_loss

    # backpropagate and update model
    loss.backward()
    self.opt_netClf.step()
    summary_stats = {'clf_loss': clf_loss*(rna_scores.size(0)+atac_scores.size(0))}
    
    #summary_stats = {'clf_loss': clf_loss*(rna_scores.size(0)+atac_scores.size(0)), 'rna_accuracy': dis_accuracy(rna_scores, rna_labels), 'rna_n_samples': rna_scores.size(0),
    #        'atac_accuracy': dis_accuracy(atac_scores, atac_labels), 'atac_n_samples': atac_scores.size(0)}
    #summary_stats={}
    return summary_stats
    
  def train(self):
    total_time=0
 
### main training loop
    for epoch in range(self.args.max_epochs):
        print('epoch',epoch)
    
        recon_rna_loss = 0
        recon_atac_loss = 0
        atac_latents_recon_loss=0
        rna_latents_recon_loss=0
        clf_loss = 0
        clf_class_loss = 0
        #AE_clf_loss = 0
        anchor_loss=0
        tri_loss=0
        MMD_loss=0
        kl_loss=0
        
        n_rna_correct = 0
        n_rna_total = 0
        n_atac_correct = 0
        n_atac_total = 0
        start_time=time.time()
        for idx, (rna_samples, atac_samples) in enumerate(zip(self.RNA_loader, self.ATAC_loader)):
            rna_inputs = rna_samples['tensor']
            atac_inputs = atac_samples['tensor']
            #print(rna_inputs.numpy())
            
            if self.args.augmentation:
                rna_signal=rna_inputs.numpy()
                atac_signal=atac_inputs.numpy()
                SNR=5
                #print(rna_signal.shape[0],rna_signal.shape[1])
                rna_noise=np.random.randn(rna_signal.shape[0],rna_signal.shape[1])
                rna_noise=rna_noise-np.mean(rna_noise)
                
                #print(atac_signal.shape[0],atac_signal.shape[1])
                atac_noise=np.random.randn(atac_signal.shape[0],atac_signal.shape[1])
                atac_noise=atac_noise-np.mean(atac_noise)
                
                rna_power=np.linalg.norm(rna_signal-rna_signal.mean())**2/rna_signal.size
                atac_power=np.linalg.norm(atac_signal-atac_signal.mean())**2/atac_signal.size
                
                rna_variance=rna_power/np.power(10,(SNR/10))
                atac_variance=atac_power/np.power(10,(SNR/10))
                
                rna_noise=(np.sqrt(rna_variance)/np.std(rna_noise))*rna_noise
                atac_noise=(np.sqrt(atac_variance)/np.std(atac_noise))*atac_noise
                
                rna_signal_noise=rna_signal+rna_noise
                atac_signal_noise=atac_signal+atac_noise
                
                rna_signal_noise=torch.from_numpy(rna_signal_noise)
                atac_signal_noise=torch.from_numpy(atac_signal_noise)
                
                rna_inputs=torch.cat((rna_inputs,rna_signal_noise.float()),0)
                atac_inputs=torch.cat((atac_inputs,atac_signal_noise.float()),0)
                
            rna_labels = rna_samples['label']
            atac_labels = atac_samples['label']
                #print(rna_labels.shape)
                #print(atac_labels.shape)
            if self.args.augmentation:
              rself.na_labels=torch.cat((rna_labels,rna_labels),0)
              atac_labels=torch.cat((atac_labels,atac_labels),0)
                  #print(rna_labels.shape)
                  #print(atac_labels.shape)
            out = self.train_autoencoders(rna_inputs, atac_inputs, rna_labels, atac_labels)

            recon_rna_loss += out['rna_recon_loss']
            recon_atac_loss += out['atac_recon_loss']
    
            clf_class_loss += out['clf_class_loss']
                
            if self.args.contrastive_loss:
                tri_loss+=out['triplet_loss']
            
            if self.args.anchor_loss:
                anchor_loss+=out['anchor_loss']
                
            if self.args.MMD_loss:
                MMD_loss+=out['MMD_loss']
                
            if self.args.consistency_loss:
                atac_latents_recon_loss+=out['atac_latents_recon_loss']
                rna_latents_recon_loss+=out['rna_latents_recon_loss']
            
            out = self.train_classifier(rna_inputs, atac_inputs)
            clf_loss += out['clf_loss']
                #n_rna_correct += out['rna_accuracy']
                #n_atac_correct += out['atac_accuracy']
        # save model
        print('clf_loss: %3f' % float(clf_loss/self.cell_num))
        print('contrastive_loss: %3f' % float(tri_loss/self.cell_num))
        print('recon_loss: %3f' % float((recon_rna_loss+recon_atac_loss)/self.cell_num))
        if epoch % self.args.save_freq == 0:
          torch.save(self.netRNA.cpu().state_dict(), os.path.join(self.args.save_dir,"netRNA_DE_%s.pth" % epoch))
          torch.save(self.netATAC.cpu().state_dict(), os.path.join(self.args.save_dir,"netATAC_DE_%s.pth" % epoch))

        end_time=time.time()
        one_epoch_time=end_time-start_time
        #print(one_epoch_time)
        total_time=total_time+one_epoch_time
        if self.args.use_gpu:
            self.netRNA.cuda()
            if self.args.discriminator:
              self.netClf.cuda()
            self.netATAC.cuda()
            if self.args.conditional:
                self.netCondClf.cuda()
    
    print('total training time:',total_time)
  def load_model(self,path1,path2):
    self.netRNA.load_state_dict(torch.load(path1))
    self.netATAC.load_state_dict(torch.load(path2))
  def test(self,RNA_test,ATAC_test):
    RNA_loader=torch.utils.data.DataLoader(RNA_test,batch_size=32,drop_last=False,shuffle=False)
    ATAC_loader=torch.utils.data.DataLoader(ATAC_test,batch_size=32,drop_last=False,shuffle=False)

    ATAC_dim=ATAC_test.data.shape[1]
    RNA_dim=RNA_test.data.shape[1]
    label_num=len(np.unique(ATAC_test.labels))

    cell_num=ATAC_test.data.shape[0]
    
    predicted_label=np.zeros(shape=(0,),dtype=int)
    ATAC_class_label=np.zeros(shape=(0,),dtype=int)
    RNA_latent=np.zeros(shape=(0,50))
    ATAC_latent=np.zeros(shape=(0,50))
    RNA_recon=np.zeros(shape=(0,RNA_dim))
    ATAC_recon=np.zeros(shape=(0,ATAC_dim))
    
    for idx, (rna_samples,atac_samples) in enumerate(zip(RNA_loader,ATAC_loader)):
      rna_inputs=rna_samples['tensor']
      rna_labels=rna_samples['label']
      atac_inputs=atac_samples['tensor']
      atac_labels=atac_samples['label'] 
      
      #print(type(rna_inputs))
      #input('pause')
      #print(ATAC_label.shape)
      #print(atac_labels.shape)
      #print(atac_labels.numpy().shape)
      ATAC_class_label=np.concatenate((ATAC_class_label,atac_labels.numpy()),axis=0)
      
      #rna_inputs=rna_inputs.cuda()
      #atac_inputs=atac_inputs.cuda()
      
      rna_latents,_=self.netRNA(rna_inputs.cuda())
      atac_latents,_=self.netATAC(atac_inputs.cuda())
      
      #_,rna_latents,_,_=netRNA(rna_inputs)
      #_,atac_latents,_,_=netATAC(atac_inputs)
      
      atac_recon=self.netATAC.decoder(atac_latents)
      rna_recon=self.netRNA.decoder(rna_latents)
      
      #print(netConClf(atac_latents))
      
      #predicted_label=np.concatenate((predicted_label,np.argmax(netConClf(atac_latents).cpu().detach().numpy(),axis=1)))
      #print(RNA_predicted_label)
      #break
      
      RNA_latent=np.concatenate((RNA_latent,rna_latents.cpu().detach().numpy()),axis=0)
      ATAC_latent=np.concatenate((ATAC_latent,atac_latents.cpu().detach().numpy()),axis=0)
      
      RNA_recon=np.concatenate((RNA_recon,rna_recon.cpu().detach().numpy()),axis=0)
      ATAC_recon=np.concatenate((ATAC_recon,atac_recon.cpu().detach().cpu().numpy()),axis=0)
    
    
    
    #print(ATAC_label.shape)
    pd.DataFrame(RNA_latent).to_csv('embedding/'+self.args.input_dir+'_rna_emb.csv')
    pd.DataFrame(ATAC_latent).to_csv('embedding/'+self.args.input_dir+'_atac_emb.csv')
    
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
    pd.DataFrame(classlist).to_csv('./embedding/'+self.args.input_dir+'_label_transferred.csv')
    #input('pause')
    #print(type(ATAC_label))
    #print(pairlist.shape)
    #print(RNA_latent.shape)
    #print(ATAC_latent.shape)
    #print(RNA_recon.shape)
    #print(ATAC_recon.shape)
    
    #sum=find_mutual_nn(RNA_latent,ATAC_latent,10,10,1)
    #print(sum)
      
    
