import torch
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator


class RebarGAN_G(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, temperature, gpu=False):
        super(RebarGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'rebargan'

        self.temperature = temperature
        # TODO(ethanjiang) switch to tunable temperature by uncommenting below code
        # self.temperature = torch.tensor(temperature, requires_grad=True).float()
        # if gpu:
        #     self.temperature = self.temperature.cuda()

        # θ parameter in the REBAR equation. It is the softmax probability of the Generator output.
        self.theta = torch.zeros(self.max_seq_len, self.vocab_size).long()


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


    def set_temperature_gradient(self, temperature_grad):
        '''Sets temperature gradient.'''

        assert self.theta.shape == temperature_grad.shape, 'temperature_grad has different shape with self.temperature'
        self.temperature.grad = temperature_grad


    def sample_theta(self, start_letter=cfg.start_letter):
        """
        Samples the network and returns θ

        :param start_letter: index of start_token
        :return θ: max_seq_length * vocab_size
        """

        theta_logit = torch.zeros(self.max_seq_len, self.vocab_size).long()
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

        :param estimated_gradient: estimated gradient for theta with respect to the reward. Shape: seq_len * vocab_size
        :return loss: REBAR loss
        """
        assert self.theta.shape == estimated_gradient.shape, 'estimated_gradient has different shape with self.theta'

        rebar_loss_matrix = self.theta * -estimated_gradient
        rebar_loss = rebar_loss_matrix.sum()
        return rebar_loss
