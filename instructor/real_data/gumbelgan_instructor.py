import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from metrics.bleu import BLEU
from models.GumbelGAN_D import GumbelGAN_D, GumbelGAN_D2
from models.GumbelGAN_G import GumbelGAN_G
from utils.true_gradient_estimator import TrueGradientEstimator
from utils.data_loader import GenDataIter, DisDataIter
from utils.helpers import get_fixed_temperature, get_losses, get_gradient_variance
from utils.text_process import tensor_to_tokens


class GumbelGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(GumbelGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = GumbelGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx,
                               gpu=cfg.CUDA)
        self.dis = GumbelGAN_D2(cfg.dis_embed_dim, cfg.max_seq_len, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_pretrain_criterion = nn.BCEWithLogitsLoss()

        # DataLoader
        self.gen_data = GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size))
        self.dis_data = DisDataIter(self.train_data.random_batch()['target'], self.gen_data.random_batch()['target'])

        # Metrics
        bleu_gram = list(range(2, cfg.max_seq_len + 1)) if cfg.max_seq_len < 5 else [2, 3, 4, 5]
        self.bleu = BLEU(test_text=tensor_to_tokens(self.gen_data.target, self.index_word_dict),
                         real_text=tensor_to_tokens(self.test_data.target, self.test_data.index_word_dict),
                         gram=bleu_gram)
        self.self_bleu = BLEU(test_text=tensor_to_tokens(self.gen_data.target, self.index_word_dict),
                              real_text=tensor_to_tokens(self.gen_data.target, self.index_word_dict),
                              gram=3)

    def _run(self):
        # =====PRE-TRAINING (GENERATOR)=====
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pretrain_generator: {}'.format(cfg.pretrained_gen_path))

        # =====PRE-TRAINING (DISCRIMINATOR)=====
        if not cfg.dis_pretrain:
            self.log.info('Starting Discriminator Training...')
            self.pretrain_discriminator(cfg.d_step, cfg.d_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
                print('Save pretrain_generator discriminator: {}'.format(cfg.pretrained_dis_path))


        # =====ADVERSARIAL TRAINING=====
        self.log.info('Starting Adversarial Training...')

        for adv_epoch in range(cfg.ADV_train_epoch):
            if adv_epoch % cfg.adv_log_step == 0:
                self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
            self.sig.update()
            if self.sig.adv_sig:
                self.adv_train_generator(cfg.ADV_g_step, adv_epoch)  # Generator
                self.adv_train_discriminator(cfg.ADV_d_step, adv_epoch)  # Discriminator
                self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature

                if adv_epoch % cfg.adv_log_step == 0:
                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                break

    def _test(self):
        print('>>> Begin test...')

        self._run()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                # =====Train=====
                pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)

                # =====Test=====
                if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                    self.log.info('[MLE-GEN] epoch %d : pre_loss = %.4f' % (epoch, pre_loss))
                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break
        if cfg.if_save and not cfg.if_test:
            self._save('MLE', epoch)

    def adv_train_generator(self, g_step, adv_epoch):
        # true_ge = TrueGradientEstimator()  TODO

        total_loss = 0
        for step in range(g_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # =====Train=====
            # vanilla_theta = self.gen.sample_vanilla_theta()
            # true_ge = true_ge.estimate_gradient(vanilla_theta...)  TODO

            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            theta_gradient = self.optimize(self.gen_adv_opt, g_loss, self.gen,
                                           theta_gradient_fetcher=self.gen.get_theta_gradient)
            theta_gradient_log_var = get_gradient_variance(theta_gradient)
            total_loss += g_loss.item()

        # =====Test=====
        avg_loss = total_loss / g_step if g_step != 0 else 0
        if adv_epoch % cfg.adv_log_step == 0:
            self.log.info('[ADV-GEN] g_loss = %.4f, temperature = %.4f, theta_gradient_log_var = %.4f'
                      % (avg_loss, self.gen.temperature, theta_gradient_log_var))


    def adv_train_discriminator(self, d_step, adv_epoch):
        total_loss = 0
        total_acc = 0
        for step in range(d_step):
            # train discriminator on a random batch
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # =====Train=====
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            _, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.dis_opt, d_loss, self.dis)

            total_loss += d_loss.item()
            predictions = torch.cat((d_out_real, d_out_fake))
            labels = torch.cat((torch.ones_like(d_out_real), torch.zeros_like(d_out_fake)))
            total_acc += torch.sum(((predictions > 0).float() == labels)).item()

            # train discriminator on a entire epoch
            # real_samples = self.train_data.target
            # gen_samples = self.gen.sample(cfg.samples_num, cfg.batch_size)
            # if cfg.CUDA:
            #     real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            # real_samples = GenDataIter(real_samples)
            # gen_samples = GenDataIter(gen_samples)
            #
            # # =====Train=====
            # for i, (real_sample_batch, gen_sample_batch) in enumerate(zip(real_samples.loader, gen_samples.loader)):
            #     real_sample_batch = F.one_hot(real_sample_batch["target"], cfg.vocab_size).float()
            #     gen_sample_batch = F.one_hot(gen_sample_batch["target"], cfg.vocab_size).float()
            #     d_out_real = self.dis(real_sample_batch)
            #     d_out_fake = self.dis(gen_sample_batch)
            #     _, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)
            #
            #     self.optimize(self.dis_opt, d_loss, self.dis)
            #
            #     total_loss += d_loss.item()
            #     predictions = torch.cat((d_out_real, d_out_fake))
            #     labels = torch.cat((torch.ones_like(d_out_real), torch.zeros_like(d_out_fake)))
            #     total_acc += torch.sum(((predictions > 0).float() == labels)).item()

        # =====Test=====
        avg_loss = total_loss / d_step if d_step != 0 else 0
        avg_acc = total_acc / (d_step * cfg.batch_size * 2) if d_step != 0 else 0
        if adv_epoch % cfg.adv_log_step == 0:
            self.log.info('[ADV-DIS] d_loss = %.4f, train_acc = %.4f,' % (avg_loss, avg_acc))

        # num_batch = cfg.samples_num // cfg.batch_size + 1
        # avg_loss = total_loss / (d_step * num_batch) if d_step != 0 else 0
        # avg_acc = total_acc / (d_step * cfg.samples_num * 2) if d_step != 0 else 0

        if adv_epoch % cfg.adv_log_step == 0:
            self.log.info('[ADV-DIS] d_loss = %.4f, train_acc = %.4f,' % (avg_loss, avg_acc))

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False, theta_gradient_fetcher=None):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        theta_gradient = None if theta_gradient_fetcher is None else theta_gradient_fetcher()
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
        return theta_gradient

    def pretrain_discriminator(self, d_step, d_epoch, phrase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        for step in range(d_step):
            # prepare loader for training
            pos_samples = self.train_data.target
            neg_samples = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
            self.dis_data.reset(pos_samples, neg_samples)

            for epoch in range(d_epoch):
                # =====Train=====
                d_loss, train_acc = self.train_dis_epoch(self.dis, self.dis_data.loader, self.dis_pretrain_criterion,
                                                         self.dis_opt, one_hot=True)

            # =====Test=====
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f,' % (
                phrase, step, d_loss, train_acc))
