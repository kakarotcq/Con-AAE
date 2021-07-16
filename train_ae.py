import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable

from dataloader import RNA_Dataset
from dataloader import ATAC_Dataset
from model import FC_Autoencoder, FC_Classifier, FC_VAE, Simple_Classifier,TripletLoss

import os
import argparse
import numpy as np
import imageio

torch.manual_seed(1)

#============ PARSE ARGUMENTS =============

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir")
    options.add_argument('--save-freq', action="store", dest="save_freq", default=500, type=int)
    options.add_argument('--pretrained-file', action="store")

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=32, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-4, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-4, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=4100, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)
    options.add_argument('--train-imagenet', action="store_true")
    options.add_argument('--conditional', action="store_true")
    #options.add_argument('--conditional-adv', action="store_true")
    options.add_argument('--triplet-loss',action="store_true")
    options.add_argument('--consistency-loss',action="store_true")
    options.add_argument('--anchor-loss',action="store_true")
    options.add_argument('--MMD-loss',action="store_true")
    options.add_argument('--VAE',action="store_true")
    options.add_argument('--discriminator',action="store_true")
    options.add_argument('--augmentation',action="store_true")

    # hyperparameters
    options.add_argument('--alpha', action="store", default=10.0, type=float)
    options.add_argument('--beta', action="store", default=1., type=float)
    options.add_argument('--beta1', action="store", default=0.5, type=float)
    options.add_argument('--beta2', action="store", default=0.999, type=float)
    options.add_argument('--lamb', action="store", default=0.00000001, type=float)
    options.add_argument('--latent-dims', action="store", default=50, type=int)

    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_false", dest="use_gpu")

    return options.parse_args()


args = setup_args()
if not torch.cuda.is_available():
    args.use_gpu = False

os.makedirs(args.save_dir, exist_ok=True)

#============= TRAINING INITIALIZATION ==============

# initialize autoencoder

if args.VAE:
  netRNA=FC_VAE(n_input=2613,n_hidden=2613)
  netATAC=FC_VAE(n_input=815,n_hidden=815)
else:
  netRNA = FC_Autoencoder(n_input=2613, nz=args.latent_dims,n_hidden=2613)
  netATAC = FC_Autoencoder(n_input=815, nz=args.latent_dims,n_hidden=815)

print("Pre-trained model loaded from %s" % args.pretrained_file)

'''if args.conditional_adv: 
    netClf = FC_Classifier(nz=args.latent_dims+10)
    assert(not args.conditional)
else:
    netClf = FC_Classifier(nz=args.latent_dims)'''

if args.conditional:
    netCondClf = Simple_Classifier(nz=args.latent_dims)
    
if args.discriminator:
    netClf=FC_Classifier(nz=args.latent_dims)

if args.use_gpu:
    netRNA.cuda()
    netATAC.cuda()
    #netClf.cuda()
    if args.conditional:
        netCondClf.cuda()
    if args.discriminator:
        netClf.cuda()

# load data
genomics_dataset = RNA_Dataset(datadir="mydata/",mode='train')
ATAC_dataset = ATAC_Dataset(datadir="mydata/",mode='train')

shuffle=False

'''if args.triplet_loss:
    shuffle=True'''

ATAC_loader = torch.utils.data.DataLoader(ATAC_dataset, batch_size=args.batch_size, drop_last=False, shuffle=shuffle)
genomics_loader = torch.utils.data.DataLoader(genomics_dataset, batch_size=args.batch_size, drop_last=False, shuffle=shuffle)

# setup optimizer
opt_netRNA = optim.Adam(list(netRNA.parameters()), lr=args.learning_rate_AE,betas=(args.beta1,args.beta2),weight_decay=0.0001)
if args.discriminator:
  opt_netClf = optim.Adam(list(netClf.parameters()), lr=args.learning_rate_D, weight_decay=args.weight_decay)
opt_netATAC = optim.Adam(list(netATAC.parameters()), lr=args.learning_rate_AE,betas=(args.beta1,args.beta2),weight_decay=0.0001)

if args.conditional:
    opt_netCondClf = optim.Adam(list(netCondClf.parameters()), lr=args.learning_rate_AE)

# loss criteria
criterion_reconstruct = nn.MSELoss()
criterion_dis=nn.MSELoss()
anchor_loss=nn.MSELoss()
criterion_classify = nn.CrossEntropyLoss()
if args.triplet_loss:
  triplet_loss=TripletLoss()


# setup logger
with open(os.path.join(args.save_dir, 'log.txt'), 'w') as f:
    print(args, file=f)
    print(netRNA, file=f)
    print(netATAC, file=f)
    #print(netClf, file=f)
    if args.conditional:
        print(netCondClf, file=f)
    if args.discriminator:
        print(netClf,file=f)

# define helper train functions

def compute_KL_loss(mu, logvar):
    if args.lamb>0:
        KLloss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return args.lamb * KLloss
    return 0

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

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

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


def train_autoencoders(rna_inputs, atac_inputs, rna_class_labels=None, atac_class_labels=None):
   
    netRNA.train()  
    netATAC.train()

    if args.discriminator:
        netClf.eval()
    if args.conditional:
        netCondClf.train()
    
    # process input data
    rna_inputs, atac_inputs = Variable(rna_inputs), Variable(atac_inputs)

    if args.use_gpu:
        rna_inputs, atac_inputs = rna_inputs.cuda(), atac_inputs.cuda()

    # reset parameter gradients
    netRNA.zero_grad()
    netATAC.zero_grad()
    if args.conditional:
      netCondClf.zero_grad()

    # forward pass
    
    if args.VAE:
      rna_recon, rna_latents, rna_mu, rna_logvar = netRNA(rna_inputs)
      atac_recon, atac_latents, atac_mu, atac_logvar = netATAC(atac_inputs)
    else:
      rna_latents,rna_recon=netRNA(rna_inputs)
      atac_latents,atac_recon=netATAC(atac_inputs)
    
    #print(rna_latents.shape)

    '''if args.conditional_adv:
        rna_class_labels, atac_class_labels = rna_class_labels.cuda(), atac_class_labels.cuda()
        rna_scores = netClf(torch.cat((rna_latents, rna_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
        atac_scores = netClf(torch.cat((atac_latents, atac_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
    else:
        rna_scores = netClf(rna_latents)
        atac_scores = netClf(atac_latents)

    rna_labels = torch.zeros(rna_scores.size(0),).float()
    atac_labels = torch.ones(atac_scores.size(0),).float()'''
    if args.discriminator:
        rna_scores = netClf(rna_latents)
        atac_scores = netClf(atac_latents)
        rna_scores=torch.squeeze(rna_scores,dim=1)
        atac_scores=torch.squeeze(atac_scores,dim=1)
        rna_labels = torch.zeros(rna_scores.size(0),).float()
        atac_labels = torch.ones(atac_scores.size(0),).float()
    
    if args.conditional:
        rna_class_scores = netCondClf(rna_latents)
        atac_class_scores = netCondClf(atac_latents)
    
    if args.use_gpu:
        if args.discriminator:
            rna_labels, atac_labels = rna_labels.cuda(), atac_labels.cuda() 
        if args.conditional:
            rna_class_labels, atac_class_labels = rna_class_labels.cuda(), atac_class_labels.cuda()

    # compute losses
    rna_recon_loss = criterion_reconstruct(rna_inputs, rna_recon)
    #print(rna_recon_loss.shape)
    #print(rna_recon_loss)
    atac_recon_loss = criterion_reconstruct(atac_inputs, atac_recon)
    #print(atac_recon_loss.shape)
    #print(atac_recon_loss)

    if args.VAE:
      kl_loss = compute_KL_loss(rna_mu, rna_logvar) + compute_KL_loss(atac_mu, atac_logvar)
    
    '''rna_scores=torch.squeeze(rna_scores,dim=1)
    atac_scores=torch.squeeze(atac_scores,dim=1)
    
    clf_loss = 0.5*criterion_dis(rna_scores, atac_labels) + 0.5*criterion_dis(atac_scores, rna_labels)'''
    #print(clf_loss.shape)
    #print(clf_loss)
    loss = args.alpha*(rna_recon_loss + atac_recon_loss)
    #loss = args.alpha*(rna_recon_loss + atac_recon_loss) + kl_loss
    if args.discriminator:
        clf_loss = 0.5*criterion_dis(rna_scores, atac_labels) + 0.5*criterion_dis(atac_scores, rna_labels)
        loss+=args.alpha*clf_loss
    
    if args.conditional:
        clf_class_loss = 0.5*criterion_classify(rna_class_scores, rna_class_labels) + 0.5*criterion_classify(atac_class_scores, atac_class_labels)
        loss += args.beta*clf_class_loss
    
    if args.VAE:
        loss+=kl_loss
    
    if args.triplet_loss:
        inputs=torch.cat((atac_latents,rna_latents),0)
        labels=torch.cat((atac_class_labels,rna_class_labels),0)
        tri_loss=triplet_loss(inputs,labels)
        loss+=args.alpha*tri_loss  
    
    if args.anchor_loss:
        anchor_loss=criterion_reconstruct(rna_latents,atac_latents)
        loss+=0.1*anchor_loss
        
    if args.consistency_loss:
        if args.VAE:
          mu,logvar=netRNA.encode(netRNA.decode(atac_latents))
          atac_latents_recon=netRNA.reparametrize(mu,logvar)
          mu,logvar=netATAC.encode(netATAC.decoder(rna_latents))
          rna_latents_recon=netATAC.reparametrize(mu,logvar)
        else:
          atac_latents_recon=netRNA.encoder(netRNA.decoder(atac_latents))
          rna_latents_recon=netATAC.encoder(netATAC.decoder(rna_latents))
        atac_latents_recon_loss=criterion_reconstruct(atac_latents,atac_latents_recon)
        rna_latents_recon_loss=criterion_reconstruct(rna_latents,rna_latents_recon)
        loss+=args.alpha*(atac_latents_recon_loss+rna_latents_recon_loss)
    if args.MMD_loss:
        MMD_loss=mmd_rbf(atac_latents,rna_latents)
        loss+=args.alpha*MMD_loss
    # backpropagate and update model
    
    
    
    #print(loss.shape)
    #print(loss)
    
    
    loss.backward()
    opt_netRNA.step()
    opt_netATAC.step()
    if args.conditional:
        opt_netCondClf.step()
    #summary_stats=0
    summary_stats = {'rna_recon_loss': rna_recon_loss.item()*rna_latents.size(0), 'atac_recon_loss': atac_recon_loss.item()*atac_latents.size(0)}
    if args.conditional:
        summary_stats['clf_class_loss'] = clf_class_loss.item()*(rna_latents.size(0)+atac_latents.size(0))
    if args.triplet_loss:
        summary_stats['triplet_loss']=tri_loss*rna_latents.size(0)
    if args.anchor_loss:
        summary_stats['anchor_loss']=anchor_loss*rna_latents.size(0)
    if args.consistency_loss:
        summary_stats['atac_latents_recon_loss']=atac_latents_recon_loss*atac_latents.size(0)
        summary_stats['rna_latents_recon_loss']=rna_latents_recon_loss*rna_latents.size(0)
    if args.MMD_loss:
        summary_stats['MMD_loss']=MMD_loss*rna_latents.size(0)
    if args.VAE:
        summary_stats['kl_loss']=kl_loss*rna_latents.size(0)
    if args.discriminator:
        summary_stats['clf_loss']=clf_loss.item()*(rna_latents.size(0)+atac_latents.size(0))

    return summary_stats

def train_classifier(rna_inputs, atac_inputs, rna_class_labels=None, atac_class_labels=None):
     
    netRNA.eval()
    netATAC.eval()
    netClf.train()

    # process input data
    rna_inputs, atac_inputs = Variable(rna_inputs), Variable(atac_inputs)

    if args.use_gpu:
        rna_inputs, atac_inputs = rna_inputs.cuda(), atac_inputs.cuda()
    
    # reset parameter gradients
    netClf.zero_grad()

    # forward pass
    if args.VAE:
      _,rna_latents, _,_ = netRNA(rna_inputs)
      _,atac_latents, _,_ = netATAC(atac_inputs)
    else:
      rna_latents, _ = netRNA(rna_inputs)
      atac_latents, _ = netATAC(atac_inputs)

    '''if args.conditional_adv:
        rna_class_labels, atac_class_labels = rna_class_labels.cuda(), atac_class_labels.cuda()
        rna_scores = netClf(torch.cat((rna_latents, rna_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
        atac_scores = netClf(torch.cat((atac_latents, atac_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
    else:'''
    rna_scores = netClf(rna_latents)
    atac_scores = netClf(atac_latents)
    
    #print(rna_scores)
    #print(atac_scores)

    rna_labels = torch.zeros(rna_scores.size(0),).float()
    atac_labels = torch.ones(atac_scores.size(0),).float()
    
    if args.use_gpu:
        rna_labels, atac_labels = rna_labels.cuda(), atac_labels.cuda()
        
    rna_scores=torch.squeeze(rna_scores,dim=1)
    atac_scores=torch.squeeze(atac_scores,dim=1)

    # compute losses
    #print(rna_scores.shape)
    #print(rna_labels.shape)
    clf_loss = 0.5*criterion_dis(rna_scores, rna_labels) + 0.5*criterion_dis(atac_scores, atac_labels)

    loss = clf_loss

    # backpropagate and update model
    loss.backward()
    opt_netClf.step()

    summary_stats = {'clf_loss': clf_loss*(rna_scores.size(0)+atac_scores.size(0)), 'rna_accuracy': dis_accuracy(rna_scores, rna_labels), 'rna_n_samples': rna_scores.size(0),
            'atac_accuracy': dis_accuracy(atac_scores, atac_labels), 'atac_n_samples': atac_scores.size(0)}

    return summary_stats

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

 
### main training loop
for epoch in range(args.max_epochs):
    print(epoch)

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

    for idx, (rna_samples, atac_samples) in enumerate(zip(genomics_loader, ATAC_loader)):
        rna_inputs = rna_samples['tensor']
        atac_inputs = atac_samples['tensor']
        
        if args.augmentation:
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
            
            #print(rna_inputs.shape)
            #print(atac_inputs.shape)
            # check the SNR
            '''Ps_rna=(np.linalg.norm(rna_signal-rna_signal.mean()))**2
            Pn_rna=(np.linalg.norm(rna_signal-rna_signal_noise))**2
            print(10*np.log10(Ps_rna/Pn_rna))'''
            #input()
        #print(type(rna_samples))
        #print(rna_samples.keys())
        if args.conditional or args.discriminator or args.triplet_loss:
            rna_labels = rna_samples['binary_label']
            atac_labels = atac_samples['binary_label']
            #print(rna_labels.shape)
            #print(atac_labels.shape)
            if args.augmentation:
              rna_labels=torch.cat((rna_labels,rna_labels),0)
              atac_labels=torch.cat((atac_labels,atac_labels),0)
              #print(rna_labels.shape)
              #print(atac_labels.shape)
            out = train_autoencoders(rna_inputs, atac_inputs, rna_labels, atac_labels)
        else:
            out = train_autoencoders(rna_inputs, atac_inputs)
        #input()
        recon_rna_loss += out['rna_recon_loss']
        recon_atac_loss += out['atac_recon_loss']
        #AE_clf_loss += out['clf_class_loss']
        
        #anchor_loss+=out['anchor_loss']

        if args.conditional:
            clf_class_loss += out['clf_class_loss']
            
        if args.triplet_loss:
            tri_loss+=out['triplet_loss']
            
        if args.VAE:
            kl_loss+=out['kl_loss']
        
        if args.anchor_loss:
            anchor_loss+=out['anchor_loss']
            
        if args.MMD_loss:
            MMD_loss+=out['MMD_loss']
            
        if args.consistency_loss:
            atac_latents_recon_loss+=out['atac_latents_recon_loss']
            rna_latents_recon_loss+=out['rna_latents_recon_loss']
        
        '''if args.conditional_adv:
            out = train_classifier(rna_inputs, atac_inputs, rna_labels, atac_labels)
        else:'''
        if args.discriminator:
            out = train_classifier(rna_inputs, atac_inputs)
            clf_loss += out['clf_loss']
            n_rna_correct += out['rna_accuracy']
            n_atac_correct += out['atac_accuracy']
        #print(n_rna_correct)
            n_rna_total += out['rna_n_samples']
            n_atac_total += out['atac_n_samples']
        #print(n_atac_correct)
            

    recon_rna_loss /= 1791.0
    recon_atac_loss/=1791.0
    atac_latents_recon_loss/=1791.0
    rna_latents_recon_loss/=1791.0
    clf_loss /= 1791.0+1791.0
    #AE_clf_loss /= 1791.0+1791.0 
    #break
    if args.conditional:
        clf_class_loss /= 1791.0+1791.0

    with open(os.path.join(args.save_dir, 'log_DE.txt'), 'a') as f:
        print('Epoch: ', epoch, ', rna recon loss: %.8f' % float(recon_rna_loss), ', atac recon loss: %.8f' % float(recon_atac_loss),', rna latents recon loss: %.8f' % float(rna_latents_recon_loss), ', atac latents recon loss: %.8f' % float(atac_latents_recon_loss),', clf loss: %.8f' % float(clf_loss), ', clf class loss: %.8f' % float(clf_class_loss), ', clf accuracy RNA: %.4f' % float(n_rna_correct / 1791.0), ', clf accuracy ATAC: %.4f' % float(n_atac_correct / 1791.0), ', triplet loss: %.4f' % tri_loss,', anchor loss: %.4f' % float(anchor_loss),', MMD loss: %.4f' % float(MMD_loss/1791.0),', KL loss: %.4f' % float(kl_loss/1791.0), file=f)
    # save model
    if epoch % args.save_freq == 0:
        torch.save(netRNA.cpu().state_dict(), os.path.join(args.save_dir,"netRNA_DE_%s.pth" % epoch))
        torch.save(netATAC.cpu().state_dict(), os.path.join(args.save_dir,"netATAC_DE_%s.pth" % epoch))
        #torch.save(netClf.cpu().state_dict(), os.path.join(args.save_dir,"netClf_%s.pth" % epoch))
        if args.conditional:
          torch.save(netCondClf.cpu().state_dict(), os.path.join(args.save_dir,"netCondClf_%s.pth" % epoch))

    if args.use_gpu:
        netRNA.cuda()
        if args.discriminator:
          netClf.cuda()
        netATAC.cuda()
        if args.conditional:
            netCondClf.cuda()

