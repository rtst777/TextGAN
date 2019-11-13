import torch
import torch.nn.functional as F

import config as cfg
from utils.helpers import get_losses

class RebarGradientEstimator:
    def __init__(self, discriminator, batch_size, real_samples, gpu=False, num_rep=1):
        """
        A class used to estimate REBAR-based gradient for GAN-based text sequence generation problem.
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

        :param input: input that will be evaluated. Shape: batch_size * seq_len * vocab_size
        :return g_loss: the loss for the samples with respect to the true label. Shape: batch_size
        """
        d_out_real = None
        d_out_fake = self.discriminator(input)
        if cfg.loss_type == 'rsgan':
            d_out_real = self.discriminator(self.real_samples)
        g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type, reduction='none')
        if self.num_rep != 1:
            g_loss = torch.mean(g_loss.reshape(self.batch_size, self.num_rep), dim=1)
        return g_loss

    def _sample_from_uniform_distribution(self, shape):
        """
        Gets a sample from uniform distribution.

        :param shape: shape of the sample
        :return sample: a sample from uniform distribution
        """
        sample = torch.zeros(shape)
        if self.gpu:
            sample = sample.cuda()
        sample.uniform_(0, 1)
        return sample

    def _compute_z_tilde(self, b, theta, eps=1e-20):
        """
        Computes z tilde under categorical distribution by using the equations in appendix C of REBAR paper.

        :param b: index of each sample element. Shape: batch_size * seq_len
        :param theta: softmax of the network output. Shape: batch_size * seq_len * vocab_size
        :return z_tilde: logit of Gumbel-Softmax which can lead to b. Shape: batch_size * seq_len * vocab_size
        """
        v = self._sample_from_uniform_distribution(theta.size())  # Shape: batch_size * seq_len * vocab_size

        idx_k = b.repeat(theta.shape[2], 1, 1).permute(1, 2, 0)  # Shape: batch_size * seq_len * vocab_size
        v_k = v.gather(dim=2, index=idx_k) # Shape: batch_size * seq_len * vocab_size
        z_tilde_k = -torch.log(- torch.log(v_k))  # Shape: batch_size * seq_len * vocab_size
        z_tilde_not_k = -torch.log(-(torch.log(v) / theta) - torch.log(v_k))  # Shape: batch_size * seq_len * vocab_size

        z_tilde_k = -torch.log(- torch.log(v_k + eps) + eps)  # Shape: batch_size * seq_len * vocab_size
        z_tilde_not_k = -torch.log(-(torch.log(v + eps) / (theta + eps)) - torch.log(v_k + eps) + eps)  # Shape: batch_size * seq_len * vocab_size
        mask = F.one_hot(b, num_classes=cfg.vocab_size).bool()  # Shape: batch_size * seq_len * vocab_size
        z_tilde = z_tilde_not_k.masked_scatter(mask, z_tilde_k[mask])  # Shape: batch_size * seq_len * vocab_size
        return z_tilde

    def _compute_gumbel_softmax(self, z, temperature):
        """
        Computes Gumbel-Softmax

        :param z: logit of Gumbel-Softmax. Shape: batch_size * seq_len * vocab_size
        :param temperature: temperature to control Gumbel-Softmax. Shape: scalar
        :return sigma_lambda: gumbel_softmax value. Shape: batch_size * seq_len * vocab_size
        """
        sigma_lambda = F.softmax(z * temperature, dim=-1)
        return sigma_lambda

    def _compute_gradient_of_theta_wrt_f(self, theta, f_input):
        """
        Computes the gradient of theta with respect to the environment function output.

        :param theta: softmax of the network output. Shape: batch_size * seq_len * vocab_size
        :param f_input: input passed to the environment function. Shape: batch_size * seq_len * vocab_size
        :return gradient: gradient of theta w.r.t. environment function output. Shape: batch_size * seq_len * vocab_size
        """
        loss_batch = self._environment_function(f_input)
        loss_batch.sum().backward(retain_graph=True)
        gradient = theta.grad.clone().detach()
        theta.grad = torch.zeros_like(theta)
        return gradient

    def _compute_gradient_of_theta_wrt_log_pb(self, theta, b, eps=1e-20):
        """
        Computes the gradient of theta with respect to log p(b).

        :param theta: softmax of the network output. Shape: batch_size * seq_len * vocab_size
        :param b: index of each sample element. Shape: batch_size * seq_len
        :return gradient: gradient of theta w.r.t. log p(b). Shape: batch_size * seq_len * vocab_size
        """
        mask = F.one_hot(b, cfg.vocab_size).bool()  # Shape: batch_size * seq_len * vocab_size
        # TODO try theta[mask]
        log_pb = torch.log(theta[mask] + eps).sum()  # scalar
        log_pb.backward(retain_graph=True)
        gradient = theta.grad.clone().detach()
        theta.grad = torch.zeros_like(theta)
        return gradient

    def _compute_gradients_from_variance_loss(self, expected_theta_gradient, temperature, eta):
        """
        Computes the gradient from variance loss using equation (6) in RELAX paper.

        The gradients will be used to update control variate parameters for minimizing the variance of the estimator.

        :param expected_theta_gradient: estimated REBAR gradient for theta. Shape: batch_size * seq_len * vocab_size
        :param temperature: temperature to control Gumbel-Softmax. Shape: scalar
        :param eta: control variate parameter to minimize the variance of the estimator. Shape: scalar
        :return expected_temperature_gradient: gradient from variance loss w.r.t. temperature. Shape: scalar
                expected_eta_gradient: gradient from variance loss w.r.t. eta. Shape: scalar
        """
        temperature.grad = torch.zeros_like(temperature)
        eta.grad = torch.zeros_like(eta)
        temperature_loss = torch.pow(expected_theta_gradient, 2).mean()
        temperature_loss.backward(retain_graph=True)
        return temperature.grad.clone().detach(), eta.grad.clone().detach()

    def estimate_gradient(self, theta_batch, z_batch, temperature, eta):
        """
        Estimates REBAR gradient by using the equation (4) in REBAR paper.

        :param theta_batch: softmax of the network output. Shape: batch_size * seq_len * vocab_size
        :param z_batch: logit of Gumbel-Softmax. Shape: batch_size * seq_len * vocab_size
        :param temperature: temperature to control Gumbel-Softmax. Shape: scalar
        :param eta: control variate parameter to minimize the variance of the estimator. Shape: scalar
        :return expected_theta_gradient: estimated REBAR gradient for theta. Shape: batch_size * seq_len * vocab_size
                expected_temperature_gradient: gradient from variance loss w.r.t. temperature. Shape: scalar
                expected_eta_gradient: gradient from variance loss w.r.t. eta. Shape: scalar
        """
        theta_batch.retain_grad()
        b_batch = torch.argmax(z_batch, dim=-1).detach()  # Shape: batch_size * seq_len
        z_tilde_batch = self._compute_z_tilde(b_batch, theta_batch)  # Shape: batch_size * seq_len * vocab_size
        sigma_lambda_z_batch = self._compute_gumbel_softmax(z_batch,
                                                            temperature)  # Shape: batch_size * seq_len * vocab_size
        sigma_lambda_z_tilde_batch = self._compute_gumbel_softmax(z_tilde_batch,
                                                                  temperature)  # Shape: batch_size * seq_len * vocab_size

        f_H_z_batch = self._environment_function(F.one_hot(b_batch, cfg.vocab_size).float())  # Shape: batch_size
        f_sigma_lambda_z_tilde_batch = self._environment_function(sigma_lambda_z_tilde_batch)  # Shape: batch_size
        gradient_wrt_log_pb_batch = self._compute_gradient_of_theta_wrt_log_pb(theta_batch, b_batch)  # Shape: batch_size * seq_len * vocab_size
        gradient_wrt_f_sigma_lambda_z_batch = self._compute_gradient_of_theta_wrt_f(theta_batch,
                                                                                    sigma_lambda_z_batch)  # Shape: batch_size * seq_len * vocab_size
        gradient_wrt_f_sigma_lambda_z_tilde_batch = self._compute_gradient_of_theta_wrt_f(theta_batch,
                                                                                          sigma_lambda_z_tilde_batch)  # Shape: batch_size * seq_len * vocab_size

        expected_theta_gradient = (f_H_z_batch - eta * f_sigma_lambda_z_tilde_batch).reshape([self.batch_size, 1, 1]) * gradient_wrt_log_pb_batch \
                               + eta * gradient_wrt_f_sigma_lambda_z_batch \
                               - eta * gradient_wrt_f_sigma_lambda_z_tilde_batch  # Shape: batch_size * seq_len * vocab_size
        expected_temperature_gradient, expected_eta_gradient = \
            self._compute_gradients_from_variance_loss(expected_theta_gradient, temperature, eta)  # Shape: scalar

        return expected_theta_gradient.clone().detach(), expected_temperature_gradient, expected_eta_gradient
