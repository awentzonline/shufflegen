from collections import OrderedDict

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pytorch_lightning as pl
from pl_bolts.models.autoencoders import VAE
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from shufflegen.datasets.sample_images import SampleImagesDataModule
from shufflegen.models.positional import (
    LearnablePositionalEncodingCat, PositionalEncoding, PositionalEncodingCat)


class SampleGenVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vae = VAE(args.piece_size, latent_dim=args.latent_dims)
        self.xformer_feature_dims = args.xformer_dims
        pe_dims = args.pe_dims if args.pe_dims else self.xformer_feature_dims
        pe_class = dict(
            add=PositionalEncoding,
            cat=PositionalEncodingCat,
            learn=LearnablePositionalEncodingCat,
        )[args.pe]
        self.positional_encoder = pe_class(pe_dims)
        self.xformer_dims = self.xformer_feature_dims + self.positional_encoder.additional_dims
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.xformer_dims, args.heads),
            args.layers
        )
        param_dims = 1 + 1 + 1  # top, left, scale
        self.vae_to_xformer = nn.Sequential(
            nn.Linear(2 * args.latent_dims, self.xformer_feature_dims),
        )
        self.xformer_to_params = nn.Sequential(
            nn.Linear(self.xformer_feature_dims, param_dims * 2),
        )
        self.xformer_to_global = nn.Sequential(
            nn.Linear(self.xformer_feature_dims, args.latent_dims * 2),
        )
        self._steps = 0  # for pretraining

    def forward(self, pieces):
        nr, nc, bs, c, h, w = pieces.shape
        pieces = pieces.view(nr * nc * bs, c, h, w)
        p, q, z = self.encode_pieces(pieces)
        enc_pieces = normal_dist_to_vector(q)
        enc_pieces = enc_pieces.view(nr * nc, bs, 2 * self.args.latent_dims)
        transformed = self.transformer(self.positional_encoder(enc_pieces))
        transformed = self.positional_encoder.decode(transformed)
        transformed_dist_params = transformed.view(nr * nc * bs, self.args.latent_dims, 2)
        transformed_dist = vector_to_normal_dist(transformed_dist_params)
        transformed_z = transformed_dist.sample()
        pieces_dec = self.decode_pieces(transformed_z)
        return pieces_dec

    def training_step(self, batch, batch_idx):
        global_imgs, pieces, tops, lefts, scales = batch
        bs, ns, c, h, w = pieces.shape
        pieces = pieces.view(ns * bs, c, h, w)
        p, q, z = self.encode_pieces(pieces)
        recons = self.decode_pieces(z)
        recons_dist = Normal(recons, 1. / 255.)
        vae_recon_loss = -recons_dist.log_prob(pieces).mean()
        vae_divergence_loss = kl_divergence(q, p).mean() * self.args.kl_beta
        vae_loss = vae_recon_loss + vae_divergence_loss
        # early out for pretraining VAE
        if self._steps < self.args.pretrain_vae:
            self._steps += 1
            if np.random.uniform() < self.args.p_log:
                with torch.no_grad():
                    vae_recons = self.reconstruct_pieces(recons)
                    save_image(vae_recons, 'recons_vae.png')
            self.log_dict(dict(
                vae_loss=vae_loss,
                vae_recon_loss=vae_recon_loss,
                vae_divergence_loss=vae_divergence_loss,
            ))
            return OrderedDict({
                "loss": vae_loss
            })

        enc_pieces = normal_dist_to_vector(q)
        enc_pieces = enc_pieces.view(bs, ns, 2 * self.args.latent_dims).detach()
        enc_pieces = enc_pieces.permute(1, 0, 2)
        enc_pieces = enc_pieces.reshape(ns * bs, 2 * self.args.latent_dims)
        enc_pieces = self.vae_to_xformer(enc_pieces)
        enc_pieces = enc_pieces.reshape(ns, bs, self.xformer_feature_dims)
        transformed = self.transformer(self.positional_encoder(enc_pieces))
        transformed = self.positional_encoder.decode(transformed)
        transformed = transformed.permute(1, 0, 2)
        transformed = transformed.view(bs, ns, self.xformer_feature_dims)
        # recon global img loss
        _, global_img_z, _ = self.encode_pieces(global_imgs)
        global_img_z = normal_dist_to_vector(global_img_z)
        global_img_z = global_img_z.detach().repeat(ns, 1, 1)
        global_img_dist = vector_to_normal_dist(global_img_z)
        # predict global from transformer features
        p_global_img_z = self.xformer_to_global(transformed)
        p_global_img_z = p_global_img_z.view(bs * ns, self.args.latent_dims, 2)
        p_global_img_dist = vector_to_normal_dist(p_global_img_z)
        global_recon_loss = kl_divergence(p_global_img_dist, global_img_dist).mean()

        # parameter loss
        p_parameter_z = self.xformer_to_params(transformed)
        

        # if np.random.uniform() < self.args.p_log:
        #     with torch.no_grad():
        #         vae_recons = self.reconstruct_pieces(recons)
        #         save_image(vae_recons, 'recons_vae.png')
        #         if 'pieces_dec' not in locals():  # hacking around with mse_loss
        #             pieces_dec = self.decode_pieces(transformed_z)
        #         recons = self.reconstruct_pieces(pieces_dec)
        #         save_image(recons, 'recons_xform.png')
        #         recon_input = self.reconstruct_pieces(pieces)
        #         save_image(recon_input, 'recon_input.png')
        #         re_piece = pieces.view(nr, nc, bs, c, h, w)
        #         re_piece = self(re_piece)
        #         re_piece = self.reconstruct_pieces(re_piece)
        #         save_image(re_piece, 'recon_input_xform.png')

        self._steps += 1

        self.log_dict(dict(
            vae_loss=vae_loss,
            vae_recon_loss=vae_recon_loss,
            vae_divergence_loss=vae_divergence_loss,
            global_recon_loss=global_recon_loss,
        ))
        return OrderedDict({
            "loss": vae_loss + global_recon_loss
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def encode_pieces(self, pieces):
        b, c, h, w = pieces.shape
        enc_pieces = self.vae.encoder(pieces)
        mu_pieces = self.vae.fc_mu(enc_pieces)
        log_var_pieces = self.vae.fc_var(enc_pieces)
        p, q, z = self.vae.sample(mu_pieces, log_var_pieces)
        return p, q, z

    def decode_pieces(self, z):
        #z = dist.rsample()
        pieces_dec = self.vae.decoder(z)
        return pieces_dec

    def reconstruct_pieces(self, pieces):
        batch_size = self.args.batch_size
        ns = self.args.img_size // self.args.piece_size
        ps = self.args.piece_size
        img_size = self.args.img_size
        pieces = pieces.view(ns * ns, batch_size, self.args.img_channels * ps * ps)
        pieces = pieces.permute(1, 2, 0)
        folded = F.fold(pieces, output_size=(img_size, img_size), kernel_size=ps, stride=ps)
        return folded

    @classmethod
    def add_argparse_args(self, p):
        p.add_argument('img_path')
        p.add_argument('--latent-dims', default=100, type=int)
        p.add_argument('--layers', default=5, type=int)
        p.add_argument('--heads', default=4, type=int)
        p.add_argument('--lr', default=0.001, type=float)
        p.add_argument('--batch-size', default=32, type=int)
        p.add_argument('--piece-size', default=16, type=int)
        p.add_argument('--img-size', default=128, type=int)
        p.add_argument('--img-channels', default=3, type=int)
        p.add_argument('--num-workers', default=2, type=int)
        p.add_argument('--p-log', default=0.02, type=float)
        p.add_argument('--tv-loss', default=1., type=float)
        p.add_argument('--pretrain-vae', default=0, type=int)
        p.add_argument('--kl-beta', default=0.1, type=float)
        p.add_argument('--pe', default='add')
        p.add_argument('--pe-dims', default=None, type=int)
        p.add_argument('--xformer-dims', default=256, type=int)
        return p


def vector_to_normal_dist(params):
    mu, log_var = params[..., 0], params[..., 1]
    std = torch.exp(log_var / 2)
    dist = Normal(mu, std)
    return dist


def permute_normal_dist(d, perm, steps, batch_size):
    original_shape = d.loc.shape
    loc = d.loc.view(steps, batch_size, -1)[perm].view(*original_shape)
    scale = d.scale.view(steps, batch_size, -1)[perm].view(*original_shape)
    return Normal(loc, scale)


def normal_dist_to_vector(d):
    return torch.cat([d.loc.unsqueeze(-1), d.scale.unsqueeze(-1)], dim=-1).squeeze(-1)


def detach_normal_dist(n):
    return Normal(n.loc.detach(), n.scale.detach())


def total_variation_loss(img, weight):
    """https://discuss.pytorch.org/t/implement-total-variation-loss-in-pytorch/55574"""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p = SampleGenVAE.add_argparse_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    model = SampleGenVAE(args)
    dm = SampleImagesDataModule(
        args.img_path, args.piece_size,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
