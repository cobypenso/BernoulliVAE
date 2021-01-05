## Coby Penso 208254128 ##

"""VAE model
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, latent_dim,device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.criterion = nn.BCELoss(reduction='sum')
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )


    def sample(self,sample_size,mu=None,logvar=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for init with zeros
        :param logvar: z logstd, None for init with zeros
        :return:
        '''
        if mu == None:
            mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)
            
        # Sample the latent variable from normal dist
        z = self.z_sample(mu, logvar)
        # Pass z through the decoder to generate an image
        x = self.upsample(z)
        x = x.view(-1, 64, 7, 7)
        x = self.decoder(x)
        return x


    def z_sample(self, mu, logvar):
        '''
            Sample latent variable - the sampling done from normal distribution with the given params
        '''
        rand_vec = torch.randn_like(mu)
        z = mu + rand_vec * torch.sqrt(torch.exp(logvar))
        return z
        
    def loss(self, x, recon, mu, logvar):
        '''
            Loss function - the loss is build from two terms:
                - BCELoss
                - KLLoss
        '''
        BCELoss = self.criterion(recon, x)
        KLLoss = 0.5*(torch.exp(logvar) + mu**2 - logvar - 1).sum()
        
        return BCELoss + KLLoss
    
    def forward(self, x):
        '''
            Forward function - pass x (image) through the VAE (encoder and decoder)
            
            @returns - Negative ELBO
        '''
        latent = self.encoder(x)
        latent = latent.view(-1, 64*7*7)
        
        # Calculate mu and logvar
        mu_z = self.mu(latent)
        logvar_z = self.logvar(latent)
        
        # Sample from the latent distribution
        z = self.z_sample(mu_z, logvar_z)
        
        # Pass through the decoder
        x_out = self.upsample(z)
        x_out = x_out.view(-1, 64, 7, 7)
        x_out = self.decoder(x_out)
        
        # Calculate the loss
        negative_ELBO = self.loss(x, x_out, mu_z, logvar_z)
        return negative_ELBO
