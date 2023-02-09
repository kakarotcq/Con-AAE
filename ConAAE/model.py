import torch
import torch.nn as nn
from torch.autograd import Variable

# adapted from pytorch/examples/vae and ethanluoyc/pytorch-vae
 
class FC_VAE(nn.Module):
    """Fully connected variational Autoencoder"""
    def __init__(self, n_input, n_hidden, nz=50):
        super(FC_VAE, self).__init__()
        self.nz = nz
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.encoder = nn.Sequential(nn.Linear(n_input, n_hidden),
                                
                                nn.BatchNorm1d(n_hidden),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(n_hidden, 100),
                                )

        self.fc1 = nn.Linear(100, nz)
        self.fc2 = nn.Linear(100, nz)

        self.decoder = nn.Sequential(nn.Linear(nz, 100),
                                     
                                     nn.BatchNorm1d(100),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(100, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(n_hidden, n_input),
                                    )
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)
    #why reparametrize
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        #print(std.size())
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        return self.decoder(z)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z
 
    def generate(self, z):
        res = self.decode(z)
        return res

class FC_Autoencoder(nn.Module):
    """Autoencoder"""
    def __init__(self, n_input, nz, n_hidden):
        super(FC_Autoencoder, self).__init__()
        self.nz = nz
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.encoder = nn.Sequential(nn.Linear(n_input, n_hidden),
                                
                                nn.LeakyReLU(inplace=True),
                                nn.BatchNorm1d(n_hidden),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(n_hidden, 100),
                                nn.BatchNorm1d(100),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(100, nz),
                                )

        self.decoder = nn.Sequential(nn.Linear(nz, 100),
                                     
                                     nn.LeakyReLU(inplace=True),
                                     nn.BatchNorm1d(100),
                                     nn.Linear(100, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(n_hidden, n_input),
                                    )

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return encoding, decoding

class FC_Classifier(nn.Module):
    """Latent space discriminator"""
    def __init__(self, nz, n_hidden=50, n_out=1):
        super(FC_Classifier, self).__init__()
        self.nz = nz
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_hidden, 2*n_hidden),
            nn.LeakyReLU(inplace=True),
#            nn.Linear(n_hidden, n_hidden),
#            nn.ReLU(inplace=True),
 #           nn.Linear(n_hidden, n_hidden),
 #           nn.ReLU(inplace=True),
            #nn.Linear(n_hidden, n_hidden),
            #nn.ReLU(inplace=True),
            nn.Linear(2*n_hidden,n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class Simple_Classifier(nn.Module):
    """Latent space discriminator"""
    def __init__(self, nz, n_out=3):
        super(Simple_Classifier, self).__init__()
        self.nz = nz
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_out),
        )

    def forward(self, x):
        return self.net(x)

class TripletLoss(nn.Module):
    def __init__(self,margin=0.3):
      super(TripletLoss,self).__init__()
      self.margin=margin
      self.ranking_loss=nn.MarginRankingLoss(margin=margin)
      
    def forward(self,inputs,labels):
      n=inputs.size(0)
      dist=torch.pow(inputs,2).sum(dim=1,keepdim=True).expand(n,n)
      dist=dist+dist.t()
      dist.addmm_(1,-2,inputs,inputs.t())
      dist=dist.clamp(min=1e-12).sqrt()
      
      mask=labels.expand(n,n).eq(labels.expand(n,n).t())
      #print(mask.shape)
      #print(mask[0])
      dist_ap,dist_an=[],[]
      for i in range(n):
        #print(i)
        dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        dist_an.append(dist[i][mask[i]==0].min().unsqueeze(0))
      dist_ap=torch.cat(dist_ap)
      dist_an=torch.cat(dist_an)
      
      y=torch.ones_like(dist_an)
      loss=self.ranking_loss(dist_an,dist_ap,y)
      return loss