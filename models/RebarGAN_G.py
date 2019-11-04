import torch
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator


class RebarGAN_G(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, temperature, eta, gpu=False):
        super(RebarGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'rebargan'
        if gpu:
            self.temperature = torch.tensor(temperature, dtype=torch.float, device='cuda', requires_grad=True)
            self.eta = torch.tensor(eta, dtype=torch.float, device='cuda', requires_grad=True)
        else:
            self.temperature = torch.tensor(temperature, dtype=torch.float, requires_grad=True)
            # Ideally, eta(ƞ) should be computed using the equation in Appendix A of REBAR paper. However, it is
            # infeasible to implement that equation under practical situation (e.g. when the environment function is a
            # Discriminator Neural Network). Therefore, we let eta to be a learnable variable.
            self.eta = torch.tensor(eta, dtype=torch.float, requires_grad=True)
        # θ parameter in the REBAR equation. It is the softmax probability of the Generator output.
        self.theta = None


    def batchPGLoss(self, inp, target, reward):
        """
        Returns a policy gradient loss

        :param inp: batch_size x seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size x seq_len
        :param reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        :return loss: policy loss
        """

        batch_size, seq_len = inp.size()
        hidden = self.init_hidden(batch_size)

        out = self.forward(inp, hidden).view(batch_size, self.max_seq_len, self.vocab_size)
        target_onehot = F.one_hot(target, self.vocab_size).float()  # batch_size * seq_len * vocab_size
        pred = torch.sum(out * target_onehot, dim=-1)  # batch_size * seq_len
        loss = -torch.sum(pred * reward)

        return loss


    def set_variance_loss_gradients(self, temperature_grad, eta_gradient):
        """
        Sets the variance loss gradients to control variate parameters.

        :param temperature_gradient: gradient from variance loss w.r.t. temperature (has to be detached). Shape: scalar
        :param eta_gradient: gradient from variance loss w.r.t. eta (has to be detached). Shape: scalar
        """
        assert self.temperature.shape == temperature_grad.shape, 'temperature_grad has different shape with self.temperature'
        assert self.eta.shape == eta_gradient.shape, 'eta_gradient has different shape with self.eta'
        self.temperature.grad = temperature_grad
        self.eta.grad = eta_gradient


    def sample_theta(self, start_letter=cfg.start_letter):
        """
        Samples the network and returns θ

        :param start_letter: index of start_token
        :return θ: max_seq_length * vocab_size
        """
        theta_logit = torch.zeros(self.max_seq_len, self.vocab_size, dtype=torch.float)
        hidden = self.init_hidden(1)
        inp = torch.LongTensor([start_letter])
        if self.gpu:
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out, hidden = self.forward(inp, hidden, need_hidden=True)  # out: 1 * vocab_size
            next_token = torch.argmax(torch.exp(out), 1)  # next_token: 1 * 1
            theta_logit[i] = out.view(-1)
            inp = next_token.view(-1)

        self.theta = F.softmax(theta_logit, dim=-1)
        return self.theta


    def computeRebarLoss(self, estimated_gradient):
        """
        Computes the loss based on the estimated REBAR gradient

        :param estimated_gradient: estimated gradient for theta with respect to the loss (has to be detached). Shape: seq_len * vocab_size
        :return loss: REBAR loss
        """
        assert self.theta is not None and \
               self.theta.shape == estimated_gradient.shape, 'estimated_gradient has different shape with self.theta'

        rebar_loss_matrix = self.theta * estimated_gradient
        rebar_loss = rebar_loss_matrix.sum()
        return rebar_loss
