import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from .glow import Glow, ZeroConv2d, gaussian_log_p
import numpy as np
from .rcan import Group
from random import sample

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DICMOR(nn.Module):
    def __init__(self, args):
        super(DICMOR, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.MSE = MSE()

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        self.flow_l = Glow(in_channel=self.d_l, n_flow=32, n_block=1, affine=True, conv_lu=False)
        self.flow_v = Glow(in_channel=self.d_v, n_flow=32, n_block=1, affine=True, conv_lu=False)
        self.flow_a = Glow(in_channel=self.d_a, n_flow=32, n_block=1, affine=True, conv_lu=False)
        self.prior_hp = ZeroConv2d(self.d_l, self.d_l * 2)
        self.prior_p = ZeroConv2d(self.d_l, self.d_l * 2)
        self.prior_wp = ZeroConv2d(self.d_l, self.d_l * 2)
        self.prior_neutral = ZeroConv2d(self.d_l, self.d_l * 2)
        self.prior_wn = ZeroConv2d(self.d_l, self.d_l * 2)
        self.prior_n = ZeroConv2d(self.d_l, self.d_l * 2)
        self.prior_hn = ZeroConv2d(self.d_l, self.d_l * 2)

        self.rec_l = nn.Sequential(
            nn.Conv1d(self.d_l, self.d_l*2, 1),
            Group(num_channels=self.d_l*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_l*2, self.d_l, 1)
        )

        self.rec_v = nn.Sequential(
            nn.Conv1d(self.d_v, self.d_v*2, 1),
            Group(num_channels=self.d_v*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_v*2, self.d_v, 1)
        )

        self.rec_a = nn.Sequential(
            nn.Conv1d(self.d_a, self.d_a*2, 1),
            Group(num_channels=self.d_a*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_a*2, self.d_a, 1)
        )

        self.cat_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_v = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_a = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # weight for each modality
        self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
        self.weight_v = nn.Linear(2 * self.d_v, 2 * self.d_v)
        self.weight_a = nn.Linear(2 * self.d_a, 2 * self.d_a)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, label=None, num_modal=None):
        if self.use_bert:
            with torch.no_grad():
                text = self.text_model(text)

        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        # Project the textual/visual/audio features
        with torch.no_grad():
            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
            proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
            proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        conv_feat_l, conv_feat_v, conv_feat_a = proj_x_l, proj_x_v, proj_x_a

        #  normalizing flow for language
        _, logdet_l, z_outs_l = self.flow_l(proj_x_l.unsqueeze(-1))
        z_l = z_outs_l
        z_outs_l = z_outs_l[0]

        #  normalizing flow for vision
        _, logdet_v, z_outs_v = self.flow_v(proj_x_v.unsqueeze(-1))
        z_v = z_outs_v
        z_outs_v = z_outs_v[0]

        #  normalizing flow for audio
        _, logdet_a, z_outs_a = self.flow_a(proj_x_a.unsqueeze(-1))
        z_a = z_outs_a
        z_outs_a = z_outs_a[0]

        log_p_sum_l, log_p_sum_v, log_p_sum_a = 0.0, 0.0, 0.0
        for i in range(label.size(0)):
            if np.round(label[i].view(-1).cpu().detach().numpy()) == 3:  # highly positive
                zero = torch.zeros_like(z_outs_l[i]).unsqueeze(0)
                mean_hp, log_sd_hp = self.prior_hp(zero).chunk(2, 1)  # learnable mean and log_sd with zeroconv
                log_p_sum_l += gaussian_log_p(z_outs_l[i].unsqueeze(0), mean_hp, log_sd_hp).view(1, -1).sum(1)
                log_p_sum_v += gaussian_log_p(z_outs_v[i].unsqueeze(0), mean_hp, log_sd_hp).view(1, -1).sum(1)
                log_p_sum_a += gaussian_log_p(z_outs_a[i].unsqueeze(0), mean_hp, log_sd_hp).view(1, -1).sum(1)
            elif np.round(label[i].view(-1).cpu().detach().numpy()) == 2:  # positive
                zero = torch.zeros_like(z_outs_l[i]).unsqueeze(0)
                mean_p, log_sd_p = self.prior_p(zero).chunk(2, 1)  # learnable mean and log_sd with zeroconv
                log_p_sum_l += gaussian_log_p(z_outs_l[i].unsqueeze(0), mean_p, log_sd_p).view(1, -1).sum(1)
                log_p_sum_v += gaussian_log_p(z_outs_v[i].unsqueeze(0), mean_p, log_sd_p).view(1, -1).sum(1)
                log_p_sum_a += gaussian_log_p(z_outs_a[i].unsqueeze(0), mean_p, log_sd_p).view(1, -1).sum(1)
            elif np.round(label[i].view(-1).cpu().detach().numpy()) == 1:  # weakly positive
                zero = torch.zeros_like(z_outs_l[i]).unsqueeze(0)
                mean_wp, log_sd_wp = self.prior_wp(zero).chunk(2, 1)  # learnable mean and log_sd with zeroconv
                log_p_sum_l += gaussian_log_p(z_outs_l[i].unsqueeze(0), mean_wp, log_sd_wp).view(1, -1).sum(1)
                log_p_sum_v += gaussian_log_p(z_outs_v[i].unsqueeze(0), mean_wp, log_sd_wp).view(1, -1).sum(1)
                log_p_sum_a += gaussian_log_p(z_outs_a[i].unsqueeze(0), mean_wp, log_sd_wp).view(1, -1).sum(1)
            elif np.round(label[i].view(-1).cpu().detach().numpy()) == 0:  # neutral
                zero = torch.zeros_like(z_outs_l[i]).unsqueeze(0)
                mean_neu, log_sd_neu = self.prior_neutral(zero).chunk(2, 1)  # learnable mean and log_sd with zeroconv
                log_p_sum_l += gaussian_log_p(z_outs_l[i].unsqueeze(0), mean_neu, log_sd_neu).view(1, -1).sum(1)
                log_p_sum_v += gaussian_log_p(z_outs_v[i].unsqueeze(0), mean_neu, log_sd_neu).view(1, -1).sum(1)
                log_p_sum_a += gaussian_log_p(z_outs_a[i].unsqueeze(0), mean_neu, log_sd_neu).view(1, -1).sum(1)
            elif np.round(label[i].view(-1).cpu().detach().numpy()) == -1:  # weakly negitive
                zero = torch.zeros_like(z_outs_l[i]).unsqueeze(0)
                mean_wn, log_sd_wn = self.prior_wn(zero).chunk(2, 1)  # learnable mean and log_sd with zeroconv
                log_p_sum_l += gaussian_log_p(z_outs_l[i].unsqueeze(0), mean_wn, log_sd_wn).view(1, -1).sum(1)
                log_p_sum_v += gaussian_log_p(z_outs_v[i].unsqueeze(0), mean_wn, log_sd_wn).view(1, -1).sum(1)
                log_p_sum_a += gaussian_log_p(z_outs_a[i].unsqueeze(0), mean_wn, log_sd_wn).view(1, -1).sum(1)
            elif np.round(label[i].view(-1).cpu().detach().numpy()) == -2:  # negitive
                zero = torch.zeros_like(z_outs_l[i]).unsqueeze(0)
                mean_n, log_sd_n = self.prior_n(zero).chunk(2, 1)  # learnable mean and log_sd with zeroconv
                log_p_sum_l += gaussian_log_p(z_outs_l[i].unsqueeze(0), mean_n, log_sd_n).view(1, -1).sum(1)
                log_p_sum_v += gaussian_log_p(z_outs_v[i].unsqueeze(0), mean_n, log_sd_n).view(1, -1).sum(1)
                log_p_sum_a += gaussian_log_p(z_outs_a[i].unsqueeze(0), mean_n, log_sd_n).view(1, -1).sum(1)
            elif np.round(label[i].view(-1).cpu().detach().numpy()) == -3:  # highly negitive
                zero = torch.zeros_like(z_outs_l[i]).unsqueeze(0)
                mean_hn, log_sd_hn = self.prior_hn(zero).chunk(2, 1)  # learnable mean and log_sd with zeroconv
                log_p_sum_l += gaussian_log_p(z_outs_l[i].unsqueeze(0), mean_hn, log_sd_hn).view(1, -1).sum(1)
                log_p_sum_v += gaussian_log_p(z_outs_v[i].unsqueeze(0), mean_hn, log_sd_hn).view(1, -1).sum(1)
                log_p_sum_a += gaussian_log_p(z_outs_a[i].unsqueeze(0), mean_hn, log_sd_hn).view(1, -1).sum(1)


        log_p_l = logdet_l.sum() + log_p_sum_l
        log_p_l = torch.max(torch.zeros_like(log_p_l),
                            (-log_p_l / (np.log(2) * proj_x_l.size(0) * proj_x_l.size(1) * proj_x_l.size(2))))
        log_p_v = logdet_v.sum() + log_p_sum_v
        log_p_v = torch.max(torch.zeros_like(log_p_v),
                            (-log_p_v / (np.log(2) * proj_x_v.size(0) * proj_x_v.size(1) * proj_x_v.size(2))))
        log_p_a = logdet_a.sum() + log_p_sum_a
        log_p_a = torch.max(torch.zeros_like(log_p_a),
                            (-log_p_a / (np.log(2) * proj_x_a.size(0) * proj_x_a.size(1) * proj_x_a.size(2))))

        #  random select modality
        modal_idx = [0, 1, 2]  # (0:text, 1:vision, 2:audio)
        ava_modal_idx = sample(modal_idx, num_modal)  # sample available modality
        if num_modal == 1:  # one modality is available
            if ava_modal_idx[0] == 0:  # has text
                proj_x_v = self.flow_v.reverse(z_l, reconstruct=True).squeeze(-1).detach()
                proj_x_a = self.flow_a.reverse(z_l, reconstruct=True).squeeze(-1).detach()
                proj_x_v = self.rec_v(proj_x_v)
                proj_x_a = self.rec_a(proj_x_a)
                loss_rec = (self.MSE(proj_x_v, conv_feat_v.detach()) + self.MSE(proj_x_a, conv_feat_a.detach())) / 2
            elif ava_modal_idx[0] == 1:  # has video
                proj_x_a = self.flow_a.reverse(z_v, reconstruct=True).squeeze(-1).detach()
                proj_x_l = self.flow_l.reverse(z_v, reconstruct=True).squeeze(-1).detach()
                proj_x_a = self.rec_a(proj_x_a)
                proj_x_l = self.rec_l(proj_x_l)
                loss_rec = (self.MSE(proj_x_l, conv_feat_l.detach()) + self.MSE(proj_x_a, conv_feat_a.detach())) / 2
            else:  # has audio
                proj_x_v = self.flow_v.reverse(z_a, reconstruct=True).squeeze(-1).detach()
                proj_x_l = self.flow_l.reverse(z_a, reconstruct=True).squeeze(-1).detach()
                proj_x_v = self.rec_v(proj_x_v)
                proj_x_l = self.rec_l(proj_x_l)
                loss_rec = (self.MSE(proj_x_l, conv_feat_l.detach()) + self.MSE(proj_x_v, conv_feat_v.detach())) / 2
        if num_modal == 2:  # two modalities are available
            if set(modal_idx)-set(ava_modal_idx) == {0}:  # L is missing (V,A available)
                l_1 = self.flow_l.reverse(z_v, reconstruct=True).squeeze(-1).detach()
                l_2 = self.flow_l.reverse(z_a, reconstruct=True).squeeze(-1).detach()
                proj_x_l = self.cat_l(torch.cat([l_1, l_2], dim=1))
                proj_x_l = self.rec_l(proj_x_l)
                loss_rec = self.MSE(proj_x_l, conv_feat_l.detach())
            if set(modal_idx)-set(ava_modal_idx) == {1}:  # V is missing (L,A available)
                v_1 = self.flow_v.reverse(z_l, reconstruct=True).squeeze(-1).detach()
                v_2 = self.flow_v.reverse(z_a, reconstruct=True).squeeze(-1).detach()
                proj_x_v = self.cat_v(torch.cat([v_1, v_2], dim=1))
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_v, conv_feat_v.detach())
            if set(modal_idx)-set(ava_modal_idx) == {2}:  # A is missing (L,V available)
                a_1 = self.flow_a.reverse(z_l, reconstruct=True).squeeze(-1).detach()
                a_2 = self.flow_a.reverse(z_v, reconstruct=True).squeeze(-1).detach()
                proj_x_a = self.cat_a(torch.cat([a_1, a_2], dim=1))
                proj_x_a = self.rec_a(proj_x_a)
                loss_rec = self.MSE(proj_x_a, conv_feat_a.detach())
        if num_modal == 3:  # no missing
            loss_rec = torch.tensor(0)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        # A residual blocka
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        res = {
            'ava_modal_idx': ava_modal_idx,
            'log_p_l': log_p_l,
            'log_p_v': log_p_v,
            'log_p_a': log_p_a,
            'loss_rec': loss_rec,
            'M': output
        }
        return res