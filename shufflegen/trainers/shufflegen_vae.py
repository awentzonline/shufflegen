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

from shufflegen.datasets.just_images import JustImagesDataModule
from shufflegen.models.positional import PositionalEncoding


class ShuffleGenVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vae = VAE(args.piece_size, latent_dim=args.latent_dims)
        args.xformer_dims = 2 * args.latent_dims
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(args.xformer_dims, args.heads),
            args.layers
        )
        self.positional_encoder = PositionalEncoding(args.xformer_dims)
        self._steps = 0  # for pretraining

    def forward(self, pieces):
        nr, nc, bs, c, h, w = pieces.shape
        pieces = pieces.view(nr * nc * bs, c, h, w)
        p, q, z = self.encode_pieces(pieces)
        enc_pieces = normal_dist_to_vector(q)
        enc_pieces = enc_pieces.view(nr * nc, bs, 2 * self.args.latent_dims)
        transformed = self.transformer(self.positional_encoder(enc_pieces))
        transformed_dist_params = transformed.view(nr * nc * bs, self.args.latent_dims, 2)
        transformed_dist = vector_to_normal_dist(transformed_dist_params)
        transformed_z = transformed_dist.sample()
        pieces_dec = self.decode_pieces(transformed_z)
        return pieces_dec

    def training_step(self, batch, batch_idx):
        images = batch
        pieces = self.make_pieces(images)

        nr, nc, bs, c, h, w = pieces.shape
        pieces = pieces.view(nr * nc * bs, c, h, w)
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
        enc_pieces = enc_pieces.view(nr * nc, bs, 2 * self.args.latent_dims)
        shuffled_pieces, perm = self.shuffle_pieces(enc_pieces)
        shuffled_pieces = shuffled_pieces.view(nr * nc, bs, 2 * self.args.latent_dims).detach()
        transformed = self.transformer(self.positional_encoder(shuffled_pieces))
        transformed_dist_params = transformed.view(nr * nc * bs, self.args.latent_dims, 2)
        transformed_dist = vector_to_normal_dist(transformed_dist_params)
        transformed_z = transformed_dist.rsample()

        # pieces_dec = self.decode_pieces(transformed_z)
        # recon_loss = F.mse_loss(pieces_dec, pieces)
        recon_loss = 0.
        tv_loss = 0.  # total_variation_loss(pieces_dec, self.args.tv_loss)
        vae_transformed_divergence_loss = \
            kl_divergence(p, detach_normal_dist(q)).mean() * self.args.kl_beta

        vae_loss = vae_loss + vae_transformed_divergence_loss

        if np.random.uniform() < self.args.p_log:
            with torch.no_grad():
                vae_recons = self.reconstruct_pieces(recons)
                save_image(vae_recons, 'recons_vae.png')
                if 'pieces_dec' not in locals():  # hacking around with mse_loss
                    pieces_dec = self.decode_pieces(transformed_z)
                recons = self.reconstruct_pieces(pieces_dec)
                save_image(recons, 'recons_xform.png')
                recon_input = self.reconstruct_pieces(pieces)
                save_image(recon_input, 'recon_input.png')
                re_piece = pieces.view(nr, nc, bs, c, h, w)
                re_piece = self(re_piece)
                re_piece = self.reconstruct_pieces(re_piece)
                save_image(re_piece, 'recon_input_xform.png')

        self._steps += 1

        self.log_dict(dict(
            recon_loss=recon_loss,
            vae_loss=vae_loss,
            vae_recon_loss=vae_recon_loss,
            vae_divergence_loss=vae_divergence_loss,
            vae_transformed_divergence_loss=vae_transformed_divergence_loss,
        ))
        return OrderedDict({
            "loss": recon_loss + vae_loss
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

    def make_pieces(self, imgs):
        batch_size = imgs.shape[0]
        ns = self.args.img_size // self.args.piece_size
        ps = self.args.piece_size
        pieces = F.unfold(imgs, ps, stride=ps)
        pieces = pieces.permute(-1, 0, 1).view(ns, ns, batch_size, self.args.img_channels, ps, ps).contiguous()
        return pieces

    def shuffle_pieces(self, pieces):
        # give each image its own shuffling
        # pieces = torch.stack([
        #     batch[torch.randperm(len(batch))] for  in pieces
        # ])
        perm = torch.randperm(len(pieces))
        pieces = pieces[perm]
        return pieces, perm

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
        p.add_argument('--xformer-dims', default=200, type=int)
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
    p = ShuffleGenVAE.add_argparse_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    model = ShuffleGenVAE(args)
    dm = JustImagesDataModule(
        args.img_path, args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
