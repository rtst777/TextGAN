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


    # For simplicity, I will assume the seq length is 3 first
    def sample_vanilla_theta(self, start_letter=cfg.start_letter):
        """
        Samples all possibility.

        :param start_letter: index of start_token (BOS)
        :return θ_: vocab_size * vocab_size * ... * vocab_size * max_seq_length (There are max_seq_length vocab size before the last factor)
        """
        

        # define the size for theta_ as a list
        the_size = torch.zeros(self.max_seq_len+1)
        the_size[:] = self.vocab_size
        the_size[the_size.shape[0]-1] = self.max_seq_len
        theta_ = torch.zeros(the_size.int().tolist())
        
        # initialize
        hidden = self.init_hidden(1)
        inp = torch.LongTensor([start_letter]*1)
        if self.gpu:
            inp = inp.cuda()
        
        # get the distribution for first word
        # and store the prob for each word
        emb = self.embeddings(inp).unsqueeze(1)  # embedding_dim
        out, hidden = self.lstm(emb, hidden)
        out = self.lstm2out(out)  # vocab_size
        out = F.softmax(out, dim=-1)  # vocab_size
        gumbel_t = self.add_gumbel(out) # vocab_size

        # due to restriction of lstm, we have to finish sequence by sequence
        def inner_embedding(n, hidden):
            inp = torch.LongTensor([n])
            emb = self.embeddings(inp).unsqueeze(1)
            out, hidden = self.lstm(emb, hidden)
            out = self.lstm2out(out)  # vocab_size
            out = F.softmax(out, dim=-1)  # vocab_size
            gumbel_t = self.add_gumbel(out) # vocab_size
            return out, hidden

        # get the distribution for second word and third word
        # and store the prob for each word "based on the previous word"
        for i in range(self.vocab_size):
            out = out.reshape(1, self.vocab_size)
            theta_[i, :, :, 0] = out[0,i]
            out, hidden = inner_embedding(i, hidden)
            for j in range(self.vocab_size):
                out = out.reshape(1, self.vocab_size)
                theta_[i,j,:,1] = out[0, j]
                out, hidden = inner_embedding(j, hidden)
                for k in range(self.vocab_size):
                    out = out.reshape(1, self.vocab_size)
                    theta_[i,j,k,2] = out[0,k]

        return theta_ #should be vocab_size*vocab_size*voc_size*3 currently




    @staticmethod
    def add_gumbel(theta, eps=1e-10, gpu=cfg.CUDA):
        u = torch.zeros(theta.size())
        if gpu:
            u = u.cuda()

        u.uniform_(0, 1)
        # F.softmax(theta_logit, dim=-1) converts theta_logit to categorical distribution.
        gumbel_t = torch.log(theta + eps) - torch.log(-torch.log(u + eps) + eps)
        return gumbel_t
