import torch
import torch.nn.functional as F

import config as cfg
from itertools import combinations_with_replacement
from utils.helpers import get_losses

class TrueGradientEstimator:
    def __init__(self, discriminator, real_samples, gpu=False, num_rep=1):
        """
        A class used to compute the ground truth gradient via exhaustive enumerating and reinforce trick
        ...

        Attributes
        ----------
        discriminator : torch.nn.Module
            Discriminator Neural Network to represent the environment function
        batch_size : int
            batch size of the sequence
        real_samples : torch.tensor
            real sequence samples from training dataset. Shape: batch_size * seq_len
        eta : float
            hyperparameter to minimize the variance of the REBAR estimator
        gpu : bool
            if CUDA is enabled
        """
        self.discriminator = discriminator
        self.real_samples = F.one_hot(real_samples, cfg.vocab_size).float()
        self.num_rep = num_rep
        self.gpu = gpu
        if gpu:
            self.real_samples = self.real_samples.cuda()


    def _environment_function(self, input):
        """
        The environment function that computes the loss for the samples with respect to the true label.

        :param input: input that will be evaluated. Shape: seq_length * vocab_size
        :return g_loss: the loss for the samples with respect to the true label. Shape: "1"
        """
        d_out_real = None
        d_out_fake = self.discriminator(input)
        if cfg.loss_type == 'rsgan':
            d_out_real = self.discriminator(self.real_samples)
        g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type, reduction='none')
        if self.num_rep != 1:
            g_loss = torch.mean(g_loss.reshape(self.batch_size, self.num_rep), dim=1)
        return g_loss


    def estimate_gradient(self, theta_, eps=1e-20):
        """
        The function to compute ground truth gradient with reinforce trick 
        (dE[f(b)]/dθ) = sigma_(f(b) * p(b) * dlog(p(b))/dθ)
        Assume the length of sentence is 2, we have 2 thetas.

        :param theta_: vanilla theta for all possible sentences. shape: vocab_size * vocab_size *...* max_seq_length
        :return gradient: the gradient matrix respct to every theta 
        """
        theta_size = theta_.shape;
        gradients = torch.zeros(theta_size[3], theta_size[0])
        # gradients = torch.zeros(theta_size[5], theta_size[0])
        # gradients = torch.zeros(theta_size[6], theta_size[0])
        

        for i,j,k in combinations_with_replacement([0,1,2], 3):

            theta_.retain_grad()
            Db = self._environment_function(F.one_hot(torch.tensor([i,j,k]).reshape(1,3), 
                 cfg.vocab_size).float()) 
            # Db = self.discriminator(F.one_hot(torch.tensor([i,j,k]).reshape(1,3), 
            #      cfg.vocab_size).float())                          
            
            log_pb = torch.log(theta_[i,j,k,0]+eps)+torch.log(theta_[i,j,k,1]+eps)\
                        +torch.log(theta_[i,j,k,2]+eps) # scalar
            log_pb.backward(retain_graph=True)
            
            # import pdb; pdb.set_trace()
            a = theta_[i,j,k,0]
            b = theta_[i,j,k,1]
            c = theta_[i,j,k,2]

            pb = a*b*c
            gradient = (theta_.grad.clone().detach())*pb*Db

            gradients[0,i] += gradient[i,j,k,0]
            gradients[1,j] += gradient[i,j,k,1]
            gradients[2,k] += gradient[i,j,k,2]

        # for i,j,k,l,m in combinations_with_replacement([0,1,2,3,4], 5):

        #     theta_.retain_grad()
        #     Db = self._environment_function(F.one_hot(torch.tensor([i,j,k,l,m]).reshape(1,5), 
        #          cfg.vocab_size).float())                         
            
        #     log_pb = torch.log(theta_[i,j,k,l,m,0]+eps)+torch.log(theta_[i,j,k,l,m,1]+eps)\
        #                 +torch.log(theta_[i,j,k,l,m,2]+eps)+torch.log(theta_[i,j,k,l,m,3]+eps)\
        #                 +torch.log(theta_[i,j,k,l,m,4]+eps)# scalar
        #     log_pb.backward(retain_graph=True)
            
        #     # import pdb; pdb.set_trace()
        #     a = theta_[i,j,k,l,m,0]
        #     b = theta_[i,j,k,l,m,1]
        #     c = theta_[i,j,k,l,m,2]
        #     d = theta_[i,j,k,l,m,3]
        #     e = theta_[i,j,k,l,m,4]

        #     pb = a*b*c*d*e
        #     gradient = (theta_.grad.clone().detach())*pb*Db

        #     gradients[0,i] += gradient[i,j,k,l,m,0]
        #     gradients[1,j] += gradient[i,j,k,l,m,1]
        #     gradients[2,k] += gradient[i,j,k,l,m,2]
        #     gradients[3,l] += gradient[i,j,k,l,m,3]
        #     gradients[4,m] += gradient[i,j,k,l,m,4]

        # for i,j,k,l,m,n in combinations_with_replacement([0,1,2,3,4,5], 6):

        #     theta_.retain_grad()
        #     Db = self._environment_function(F.one_hot(torch.tensor([i,j,k,l,m,n]).reshape(1,6), 
        #          cfg.vocab_size).float())                         
            
        #     log_pb = torch.log(theta_[i,j,k,l,m,n,0]+eps)+torch.log(theta_[i,j,k,l,m,n,1]+eps)\
        #                 +torch.log(theta_[i,j,k,l,m,n,2]+eps)+torch.log(theta_[i,j,k,l,m,n,3]+eps)\
        #                 +torch.log(theta_[i,j,k,l,m,n,4]+eps)+torch.log(theta_[i,j,k,l,m,n,5]+eps)#scaler
                        
        #     log_pb.backward(retain_graph=True)
            
        #     # import pdb; pdb.set_trace()
        #     a = theta_[i,j,k,l,m,n,0]
        #     b = theta_[i,j,k,l,m,n,1]
        #     c = theta_[i,j,k,l,m,n,2]
        #     d = theta_[i,j,k,l,m,n,3]
        #     e = theta_[i,j,k,l,m,n,4]
        #     f = theta_[i,j,k,l,m,n,5]

        #     pb = a*b*c*d*e*f
        #     gradient = (theta_.grad.clone().detach())*pb*Db

        #     gradients[0,i] += gradient[i,j,k,l,m,n,0]
        #     gradients[1,j] += gradient[i,j,k,l,m,n,1]
        #     gradients[2,k] += gradient[i,j,k,l,m,n,2]
        #     gradients[3,l] += gradient[i,j,k,l,m,n,3]
        #     gradients[4,m] += gradient[i,j,k,l,m,n,4]
        #     gradients[5,n] += gradient[i,j,k,l,m,n,5]

        return gradients