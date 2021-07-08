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


class SamplePredVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vae = VAE(args.piece_size, latent_dim=args.latent_dims)
        param_dims = 1 + 1 + 1  # top, left, scale
        decoder_input_dims = param_dims + args.latent_dims
        self.learn_dist = True

        self.patch_decoder = nn.Sequential(
            nn.Linear(decoder_input_dims, args.hidden_dims),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_dims, args.hidden_dims),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_dims, args.latent_dims * (1 + args.learn_dist)),
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
        pieces_p, pieces_q, pieces_z = self.encode_pieces(pieces)
        recons = self.decode_pieces(pieces_z)
        recons_dist = Normal(recons, 1. / 255.)
        vae_recon_loss = -recons_dist.log_prob(pieces).mean()
        vae_divergence_loss = kl_divergence(pieces_q, pieces_p).mean() * self.args.kl_beta
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

        # predict patch z from f(patch location params, global image z)
        p, global_q, global_img_z = self.encode_pieces(global_imgs)
        global_img_z = global_img_z.unsqueeze(1).repeat(1, ns, 1)
        global_img_z = global_img_z.view(bs * ns, self.args.latent_dims)
        global_vae_divergence_loss = kl_divergence(global_q, p).mean() * self.args.kl_beta
        params = torch.cat([tops[..., None], lefts[..., None], scales[..., None]], -1)
        params = params.view(bs * ns, -1)
        inputs = torch.cat([global_img_z, params], -1)
        p_patch_z = self.patch_decoder(inputs)
        if self.args.detach_pieces:
            pieces_q = detach_normal_dist(pieces_q)
        if self.args.learn_dist:
            p_patch_z = vector_to_normal_dist(p_patch_z.view(bs * ns, self.args.latent_dims, 2))
            patch_latent_loss = kl_divergence(p_patch_z, pieces_q).mean()
        else:
            patch_latent_loss = -pieces_q.log_prob(p_patch_z).mean()

        if np.random.uniform() < self.args.p_log:
            with torch.no_grad():
                save_image(recons, 'recons_vae.png')
                if 'pieces_dec' not in locals():  # hacking around with mse_loss
                    if self.args.learn_dist:
                        p_patch_z = p_patch_z.sample()
                    pieces_dec = self.decode_pieces(p_patch_z)
                save_image(pieces_dec, 'recons_xform.png')
                save_image(pieces, 'recon_input.png')

        self._steps += 1

        self.log_dict(dict(
            vae_loss=vae_loss,
            vae_recon_loss=vae_recon_loss,
            patch_latent_loss=patch_latent_loss,
            global_vae_divergence_loss=global_vae_divergence_loss,
        ))
        return OrderedDict({
            "loss": vae_loss + patch_latent_loss + global_vae_divergence_loss
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

    def render_patches(self, global_zs, size):
        idxs = torch.arange(size)
        xs = idxs / size
        lefts = xs[None].repeat(size, 1)
        rights = lefts.T
        scales = torch.full((size * size), 1. / size)


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
        p.add_argument('--hidden-dims', default=512, type=int)
        p.add_argument('--learn-dist', action='store_true')
        p.add_argument('--detach-pieces', action='store_true')
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
    p = SamplePredVAE.add_argparse_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    model = SamplePredVAE(args)
    dm = SampleImagesDataModule(
        args.img_path, args.piece_size,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
