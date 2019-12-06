import torch
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator

from itertools import combinations_with_replacement


class GumbelGAN_G(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(GumbelGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)

        self.name = 'gumbelgan'
        self.temperature = 1.0  # init value is 1.0
        self.init_params()
        self.theta = None

    def step(self, inp, hidden, i):
        """
        GumbelGAN step forward
        :param inp: [batch_size]
        :param hidden: (h, c)
        :param i: sequence index
        :return: pred, hidden, next_token, next_token_onehot
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
        """
        emb = self.embeddings(inp).unsqueeze(1)
        out, hidden = self.lstm(emb, hidden)
        out = self.lstm2out(out.squeeze(1))
        out = F.softmax(out, dim=-1)
        out.retain_grad()
        self.theta.append(out)
        gumbel_t = self.add_gumbel(out)
        next_token = torch.argmax(gumbel_t, dim=1).detach()
        # next_token_onehot = F.one_hot(next_token, cfg.vocab_size).float()  # not used yet
        next_token_onehot = None

        pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size

        return pred, hidden, next_token, next_token_onehot

    def sample(self, num_samples, batch_size, one_hot=False, start_letter=cfg.start_letter):
        """
        Sample from GumbelGAN Generator
        - one_hot: if return pred of GumbelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        self.theta = []

        if one_hot:
            all_preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)
            if self.gpu:
                all_preds = all_preds.cuda()

        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                pred, hidden, next_token, _ = self.step(inp, hidden, i)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                if one_hot:
                    all_preds[:, i] = pred
                inp = next_token
        samples = samples[:num_samples]  # num_samples * seq_len

        if one_hot:
            return all_preds  # batch_size * seq_len * vocab_size
        return samples

    def get_theta_gradient(self, batch_size=cfg.batch_size):
        assert self.theta is not None, 'theta gradient is not available in GumbelGAN'

        theta_grad = torch.zeros(batch_size, self.max_seq_len, self.vocab_size, dtype=torch.float)
        for i, theta_i in enumerate(self.theta):
            grad = self.theta[i].grad.clone().detach()
            assert grad is not None, 'theta gradient is not available in GumbelGAN'
            theta_grad[:, i, :] = grad

        if self.gpu:
            theta_grad = theta_grad.cuda()
        return theta_grad


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

        # Len 3
        for i,j,k in combinations_with_replacement([0,1,2], 3):
            out = out.reshape(1, self.vocab_size)
            theta_[i,:,:,0] = out[0,i]
            out, hidden = inner_embedding(i, hidden)
            
            out = out.reshape(1, self.vocab_size)
            theta_[i,j,:,1] = out[0, j]
            out, hidden = inner_embedding(j, hidden)
                
            out = out.reshape(1, self.vocab_size)
            theta_[i,j,k,2] = out[0,k]


        # for i,j,k,l,m in combinations_with_replacement([0,1,2,3,4], 5):
        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,:,:,:,:,0] = out[0,i]
        #     out, hidden = inner_embedding(i, hidden)
            
        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,j,:,:,:,1] = out[0, j]
        #     out, hidden = inner_embedding(j, hidden)
            
        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,j,k,:,:,2] = out[0,k]
                    
        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,j,k,l,:,3] = out[0,l]
                        
        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,j,k,l,m,4] = out[0,m]

        # for i,j,k,l,m,n in combinations_with_replacement([0,1,2,3,4,5], 6):
        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,:,:,:,:,:,0] = out[0,i]
        #     out, hidden = inner_embedding(i, hidden)
            
        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,j,:,:,:,:,1] = out[0, j]
        #     out, hidden = inner_embedding(j, hidden)
            
        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,j,k,:,:,:,2] = out[0,k]
                    
        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,j,k,l,:,:,3] = out[0,l]
                        
        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,j,k,l,m,:,4] = out[0,m]

        #     out = out.reshape(1, self.vocab_size)
        #     theta_[i,j,k,l,m,n,5] = out[0,n]

        return theta_ 


    @staticmethod
    def add_gumbel(theta, eps=1e-10, gpu=cfg.CUDA):
        u = torch.zeros(theta.size())
        if gpu:
            u = u.cuda()

        u.uniform_(0, 1)
        # F.softmax(theta_logit, dim=-1) converts theta_logit to categorical distribution.
        gumbel_t = torch.log(theta + eps) - torch.log(-torch.log(u + eps) + eps)
        # gumbel_t = theta_logit - torch.log(-torch.log(u + eps) + eps)  TODO(ethanjiang) should come back to this formula and evaluate again
        return gumbel_t
