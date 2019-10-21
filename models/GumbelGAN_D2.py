import torch.nn as nn

from models.discriminator import CNNDiscriminator

dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]


class GumbelGAN_D(CNNDiscriminator):
    def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size, padding_idx, gpu=False, dropout=0.25):
        super(GumbelGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx,
                                       gpu, dropout)

        self.feature2out = nn.Linear(self.feature_dim, 1)  # For 'rsgan' loss
        # self.feature2out = nn.Linear(self.feature_dim, 2)  # For 'JS' loss
