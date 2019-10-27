import torch
import torch.nn as nn
import torch.nn.functional as F


class RebarGradientEstimator:
    def __init__(self, discriminator, batch_size, is_multiclass=True, gpu=False, true_label=1):
        self.discriminator = discriminator
        if is_multiclass:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.true_label = torch.tensor(true_label)
        self.batch_size = batch_size
        self.gpu = gpu

    def _environment_function(self, input):
        """
        The environment function that computes the loss for the samples with respect to the true label.

        :param input: input that will be evaluated. Shape: batch_size * seq_len * vocab_size
        :return loss: the loss for the samples with respect to the true label. Shape: batch_size
        """
        flat_input = input.view(-1, input.shape[-1])
        target = self.true_label.repeat(flat_input.shape[0])
        loss = self.criterion(flat_input, target)
        return loss

    def _compute_eta(self):
        """
        Computes eta.

        :return ƞ
        """
        # TODO
        eta = 1
        return eta

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

    def _compute_gumbel_softmax_logit(self, theta, eps=1e-10):
        """
        Computes the logit of Gumbel-Softmax.

        :param theta: softmax of the network output. Shape: batch_size * seq_len * vocab_size
        :return z: logit of Gumbel-Softmax. Shape: batch_size * seq_len * vocab_size
        """
        u = self._sample_from_uniform_distribution(theta.size())
        z = torch.log(theta + eps) - torch.log(-torch.log(u + eps) + eps)
        return z

    def _compute_z_tilde(self, b, theta):
        """
        Computes z tilde under categorical distribution

        :param b: index of each sample element. Shape: batch_size * seq_len
        :param theta: softmax of the network output. Shape: batch_size * seq_len * vocab_size
        :return z_tilde: logit of Gumbel-Softmax which can lead to b. Shape: batch_size * seq_len * vocab_size
        """
        v = self._sample_from_uniform_distribution(theta.size())
        # TODO

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
        loss_batch.sum().backward()
        gradient = theta.grad.clone().detach()
        theta.grad = torch.zeros(theta.shape)
        return gradient

    def _compute_gradient_of_theta_wrt_log_pb(self, theta, b):
        """
        Computes the gradient of theta with respect to log p(b).

        :param theta: softmax of the network output. Shape: batch_size * seq_len * vocab_size
        :param b: index of each sample element. Shape: batch_size * seq_len
        :return gradient: gradient of theta w.r.t. log p(b). Shape: batch_size * seq_len * vocab_size
        """
        res = 1  # TODO
        res.backward()
        gradient = theta.grad.clone().detach()
        theta.grad = torch.zeros(theta.shape)
        return gradient

    def estimate_gradient(self, theta, temperature, f):
        """
        Estimates REBAR gradient.

        :param theta: softmax of the network output. Shape: seq_len * vocab_size
        :param temperature: temperature to control Gumbel-Softmax. Shape: scalar
        :return expected_theta_gradient: estimated REBAR gradient for theta. Shape: seq_len * vocab_size
                expected_temperature_gradient: estimated REBAR gradient for temperature. Shape: scalar
        """

        theta_batch = theta.repeat(self.batch_size, 1, 1)  # Shape: batch_size * seq_len * vocab_size
        b_batch = torch.argmax(theta_batch, dim=-1).detach()  # Shape: batch_size * seq_len
        z_batch = self._compute_gumbel_softmax_logit(theta_batch)  # Shape: batch_size * seq_len * vocab_size
        z_tilde_batch = self._compute_z_tilde(b_batch, theta_batch)  # Shape: batch_size * seq_len * vocab_size
        sigma_lambda_z_batch = self._compute_gumbel_softmax(z_batch,
                                                            temperature)  # Shape: batch_size * seq_len * vocab_size
        sigma_lambda_z_tilde_batch = self._compute_gumbel_softmax(z_tilde_batch,
                                                                  temperature)  # Shape: batch_size * seq_len * vocab_size

        f_H_z_batch = self._environment_function(b_batch)  # Shape: batch_size
        eta = self._compute_eta()  # Shape: scalar
        f_sigma_lambda_z_tilde_batch = self._environment_function(sigma_lambda_z_tilde_batch)  # Shape: batch_size
        gradient_wrt_log_pb_batch = self._compute_gradient_of_theta_wrt_log_pb(theta_batch,
                                                                               b_batch)  # Shape: batch_size * seq_len * vocab_size
        gradient_wrt_f_sigma_lambda_z_batch = self._compute_gradient_of_theta_wrt_f(theta_batch,
                                                                                    sigma_lambda_z_batch)  # Shape: batch_size * seq_len * vocab_size
        gradient_wrt_f_sigma_lambda_z_tilde_batch = self._compute_gradient_of_theta_wrt_f(theta_batch,
                                                                                          sigma_lambda_z_tilde_batch)  # Shape: batch_size * seq_len * vocab_size

        theta_gradient_batch = (f_H_z_batch - eta * f_sigma_lambda_z_tilde_batch).reshape([self.batch_size, 1, 1]) * gradient_wrt_log_pb_batch \
                               + eta * gradient_wrt_f_sigma_lambda_z_batch \
                               - eta * gradient_wrt_f_sigma_lambda_z_tilde_batch  # Shape: batch_size * seq_len * vocab_size
        expected_theta_gradient = theta_gradient_batch.mean(dim=0)  # Shape: seq_len * vocab_size
        expected_temperature_gradient = temperature.grad / self.batch_size  # Shape: scalar

        return expected_theta_gradient, expected_temperature_gradient
