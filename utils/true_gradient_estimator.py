import torch
import torch.nn.functional as F

import config as cfg
from utils.helpers import get_losses

class TrueGradientEstimator:
    def __init__(self, discriminator, batch_size, real_samples, gpu=False, num_rep=1):
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
        self.batch_size = batch_size
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


    def estimate_gradient(self, theta_):
        """
        The function to compute ground truth gradient with reinforce trick 
        (dE[f(b)]/dθ) = (1/n)*sigma_(f(b) * dlog(p(b))/dθ)
        Assume the length of sentence is 2, we have 2 thetas.

        :param theta_: vanilla theta for all possible sentences. shape: vocab_size * vocab_size * max_seq_length
        :return gradient: the gradient matrix respct to every theta 
        """
        theta_size = theta_.shape;
        num_sentences = theta_size[0]*theta_size[1]

        gradients = torch.zeros(theta_.shape[0])
        
        for i in theta.shape[0]:
            for j in theta.shape[1]:
                Db = self._environment_function(F.one_hot([i,j], cfg.vocab_size).float())
                grad_1 = 1/theta_[i]*Db
                grad_2 = 1/theta_[j]*Db
                gradients[i] += grad_1
                gradients[j] += grad_2

        return gradients/num_sentences