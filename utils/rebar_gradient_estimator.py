import torch
import torch.nn.functional as F


class RebarGradientEstimator:
    def __init__(self, gen, batch_size, gpu=True):
        self.gen = gen
        self.batch_size = batch_size
        self.gpu = gpu


    def _compute_eta(self, f):
        """
        Gets a sample from uniform distribution.

        :param f: environment function that converts samples to the reward signal
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

        :param theta: softmax of the network output. Shape: seq_len * vocab_size
        :return z: logit of Gumbel-Softmax. Shape: seq_len * vocab_size
        """
        u = self._sample_from_uniform_distribution(theta.size())
        z = torch.log(theta + eps) - torch.log(-torch.log(u + eps) + eps)
        return z


    def _compute_z_tilde(self, b, theta):
        """
        Computes z tilde under categorical distribution

        :param b: index of each sample element. Shape: seq_len
        :param theta: softmax of the network output. Shape: seq_len * vocab_size
        :return z_tilde
        """
        v = self._sample_from_uniform_distribution(theta.size())
        # TODO


    def _compute_gumbel_softmax(self, z, temperature):
        """
        Computes Gumbel-Softmax

        :param z: logit of Gumbel-Softmax. Shape: seq_len * vocab_size
        :param temperature: temperature to control Gumbel-Softmax
        :return sigma_lambda: gumbel_softmax value. Shape: seq_len * vocab_size
        """
        sigma_lambda = F.softmax(z * temperature, dim=-1)
        return sigma_lambda


    def _compute_gradient_of_theta_wrt_f(self, theta, f, parameter_of_f):
        """
        Computes the gradient of theta with respect to the environment function.

        :param theta: softmax of the network output. Shape: seq_len * vocab_size
        :param f: environment function that converts samples to the reward signal
        :param parameter_of_f: parameter passed to the environment function. Shape: seq_len * vocab_size
        :return gradient: seq_len * vocab_size
        """
        res = f(parameter_of_f)
        res.backward()
        gradient = theta.grad
        theta.grad = torch.zeros(theta.shape)
        return gradient


    def _compute_gradient_of_theta_wrt_log_pb(self, theta, b):
        """
        Computes the gradient of theta with respect to log p(b).

        :param theta: softmax of the network output. Shape: seq_len * vocab_size
        :param b: index of each sample element. Shape: seq_len
        :return gradient: seq_len * vocab_size
        """
        res = 1 # TODO
        res.backward()
        gradient = theta.grad
        theta.grad = torch.zeros(theta.shape)
        return gradient


    def estimate_gradient(self, theta, temperature, f):
        """
        Estimates REBAR gradient.

        :param theta: softmax of the network output. Shape: seq_len * vocab_size
        :param temperature: temperature to control Gumbel-Softmax
        :param f: environment function that converts samples to the reward signal
        :return gradient: estimated REBAR gradient
                temperature_grad: gradient of the temperature
        """
        b = torch.argmax(theta, dim=-1).detach()  # Shape: seq_len
        z = self._compute_gumbel_softmax_logit(theta)  # Shape: seq_len * vocab_size
        z_tilde = self._compute_z_tilde(b, theta)  # Shape: seq_len * vocab_size
        sigma_lambda_z = self._compute_gumbel_softmax(z, temperature)  # Shape: seq_len * vocab_size
        sigma_lambda_z_tilde = self._compute_gumbel_softmax(z_tilde, temperature)  # Shape: seq_len * vocab_size

        # TODO implement expected REBAR gradient (should be vectorized)

        temperature_grad = temperature.grad / self.batch_size
        pass
