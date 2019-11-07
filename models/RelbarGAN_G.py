import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator
from models.relational_rnn_general import RelationalMemory


class RelbarGAN_G(LSTMGenerator):
    def __init__(self, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
                 temperature, eta, gpu=False):
        super(RelbarGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'relbargan'
        if gpu:
            self.temperature = torch.tensor(temperature, dtype=torch.float, device='cuda', requires_grad=True)
            self.eta = torch.tensor(eta, dtype=torch.float, device='cuda', requires_grad=True)
        else:
            self.temperature = torch.tensor(temperature, dtype=torch.float, requires_grad=True)
            self.eta = torch.tensor(eta, dtype=torch.float, requires_grad=True)

        # RMC
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.hidden_dim = mem_slots * num_heads * head_size
        self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
                                     num_heads=num_heads, return_all_outputs=True)
        self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        # LSTM
        # self.hidden_dim = 512
        # self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True)
        # self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        # θ parameter in the REBAR equation. It is the softmax probability of the Generator output.
        self.theta = None

        self.init_params()

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


    def sample_theta(self, batch_size, start_letter=cfg.start_letter):
        """
        Samples the network based on Gumbel logit.

        :param start_letter: index of start_token
        :return θ: batch_size * max_seq_length * vocab_size
                z: batch_size * max_seq_length * vocab_size
        """
        self.theta = torch.zeros(batch_size, self.max_seq_len, self.vocab_size, dtype=torch.float)
        z = torch.zeros(batch_size, self.max_seq_len, self.vocab_size, dtype=torch.float)

        hidden = self.init_hidden(batch_size)
        inp = torch.LongTensor([start_letter] * batch_size)
        if self.gpu:
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * embedding_dim
            out, hidden = self.lstm(emb, hidden)
            out = self.lstm2out(out.squeeze(1))  # batch_size * vocab_size
            out = F.softmax(out, dim=-1)  # batch_size * vocab_size
            gumbel_t = self.add_gumbel(out)  # batch_size * vocab_size
            next_token = torch.argmax(gumbel_t, dim=1).detach()  # batch_size * vocab_size

            self.theta[:, i, :] = out
            z[:, i, :] = gumbel_t
            inp = next_token.view(-1)

        if self.gpu:
            self.theta = self.theta.cuda()
            z = z.cuda()
        return self.theta, z


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


    def init_hidden(self, batch_size=cfg.batch_size):
        """init RMC memory"""
        memory = self.lstm.initial_state(batch_size)
        memory = self.lstm.repackage_hidden(memory)  # detch memory at first
        return memory.cuda() if self.gpu else memory

    @staticmethod
    def add_gumbel(theta, eps=1e-10, gpu=cfg.CUDA):
        u = torch.zeros(theta.size())
        if gpu:
            u = u.cuda()

        u.uniform_(0, 1)
        # F.softmax(theta_logit, dim=-1) converts theta_logit to categorical distribution.
        gumbel_t = torch.log(theta + eps) - torch.log(-torch.log(u + eps) + eps)
        return gumbel_t
