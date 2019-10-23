from models.discriminator import CNNDiscriminator

dis_filter_sizes_raw = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters_raw = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]


class RebarGAN_D(CNNDiscriminator):
    def __init__(self, embed_dim, max_seq_len, vocab_size, padding_idx, gpu=False, dropout=0.25):
        dis_filter_sizes, dis_num_filters = CNNDiscriminator.get_filters(max_seq_len, dis_filter_sizes_raw,
                                                                         dis_num_filters_raw)
        super(RebarGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx, gpu,
                                       dropout)
