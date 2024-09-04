import os
import math

import torch
from torch import optim
import pytorch_lightning as pl
import torchmetrics
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms

from models import BaseVAE
from models.types_ import *


class ResidualVaeGanEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict, ) -> None:
        super(ResidualVaeGanEXperiment, self).__init__()
        self.automatic_optimization = False
        self.model = vae_model
        self.params = params['exp_params']

        self.margin = 0.35
        self.equilibrium = 0.68

        self.decay_mse = self.params['decay_mse']
        self.decay_margin = self.params['decay_margin']
        self.lambda_mse = self.params['lambda_mse']
        self.decay_lr = self.params['decay_lr']
        self.decay_equilibrium = self.params['decay_equilibrium']
        self.critrion = nn.BCELoss()
        self.kld_weight = self.params['kld_weight']

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        opt_e = self.optimizers()[0]
        opt_d = self.optimizers()[1]
        opt_dis = self.optimizers()[2]
        lr_e = self.lr_schedulers()[0]
        lr_d = self.lr_schedulers()[1]
        lr_dis = self.lr_schedulers()[2]
        real_img = batch['img']
        b_size = real_img.size(0)
        self.curr_device = real_img.device
        real_img.requires_grad = False
        # results=self.model(real_img)

        mu, logvar = self.model.encode(real_img)
        z_e = self.model.reparameterize(mu, logvar)
        z_d = z_e.detach()

        x_tilde_e = self.model.decode(z_e)
        x_tilde_d = self.model.decode(z_d)

        x_tilde_dis = x_tilde_e.detach()
        z_p = torch.randn(b_size, z_d.size(1)).to(self.curr_device)
        x_p = self.model.decode(z_p)
        x_p_dis = x_p.detach()

        kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        nle = F.mse_loss(x_tilde_e, real_img)
        # nld = F.mse_loss(x_tilde_d, real_img)

        disc_layer_original_e = self.model.discriminate(real_img, "REC")
        disc_layer_predicted_e = self.model.discriminate(x_tilde_e, "REC")

        disc_layer_original_d = self.model.discriminate(real_img, "REC")
        disc_layer_predicted_d = self.model.discriminate(x_tilde_d, "REC")

        disc_class_original_dis = self.model.discriminate(real_img, "GAN")
        disc_class_predicted_dis = self.model.discriminate(x_tilde_dis, "GAN")
        disc_class_sampled_dis = self.model.discriminate(x_p_dis, "GAN")

        disc_class_original_d = self.model.discriminate(real_img, "GAN")
        disc_class_predicted_d = self.model.discriminate(x_tilde_d, "GAN")
        disc_class_sampled_d = self.model.discriminate(x_p, "GAN")

        mse_e = F.mse_loss(disc_layer_predicted_e, disc_layer_original_e)
        mse_d = F.mse_loss(disc_layer_predicted_d, disc_layer_original_d)

        bce_dis_original_d = self.critrion(disc_class_original_d, torch.ones_like(disc_class_original_d))
        bce_dis_predicted_d = self.critrion(disc_class_predicted_d, torch.zeros_like(disc_class_predicted_d))
        bce_dis_sampled_d = self.critrion(disc_class_sampled_d, torch.zeros_like(disc_class_sampled_d))

        bce_dis_original_dis = self.critrion(disc_class_original_dis, torch.ones_like(disc_class_original_dis))
        bce_dis_predicted_dis = self.critrion(disc_class_predicted_dis, torch.zeros_like(disc_class_predicted_dis))
        bce_dis_sampled_dis = self.critrion(disc_class_sampled_dis, torch.zeros_like(disc_class_sampled_dis))

        loss_encoder = self.kld_weight * kl + mse_e
        loss_discriminator = bce_dis_original_dis + bce_dis_predicted_dis + bce_dis_sampled_dis
        loss_decoder = self.lambda_mse * mse_d - (1.0 - self.lambda_mse) * (
                bce_dis_original_d + bce_dis_predicted_d + bce_dis_sampled_d)
        # loss_decoder =  mse_d
        self.model.zero_grad()

        self.manual_backward(loss_encoder,
                             retain_graph=True)  # someone likes to clamp the grad here: [p.grad.data.clamp_(-1,1) for p in net.encoder.parameters()]
        opt_e.step()
        self.model.zero_grad()

        train_dis = True
        train_dec = True

        if torch.mean(bce_dis_original_dis).item() < self.equilibrium - self.margin or torch.mean(
                bce_dis_predicted_dis).item() < self.equilibrium - self.margin:
            train_dis = False
        if torch.mean(bce_dis_original_d).item() > self.equilibrium + self.margin or torch.mean(
                bce_dis_predicted_d).item() > self.equilibrium + self.margin:
            train_dec = False
        if train_dec is False and train_dis is False:
            train_dis = True
            train_dec = True
        if train_dec:
            self.manual_backward(loss_decoder,
                                 retain_graph=True)  # [p.grad.data.clamp_(-1,1) for p in net.decoder.parameters()]
            opt_d.step()
            self.model.discriminator.zero_grad()  # clean the discriminator

        if train_dis:
            self.manual_backward(
                loss_discriminator)  # [p.grad.data.clamp_(-1,1) for p in net.discriminator.parameters()]
            opt_dis.step()
            opt_dis.zero_grad()
        self.margin *= self.decay_margin
        self.equilibrium *= self.decay_equilibrium
        if self.margin > self.equilibrium:
            self.equilibrium = self.margin
        self.lambda_mse *= self.decay_mse
        if self.lambda_mse > 1:
            self.lambda_mse = 1

        self.log_dict({'lossE': loss_encoder, 'lossD': loss_decoder, 'loss_Dis': loss_discriminator, 'real_kl': kl,
                       'loss_rec': nle, 'loss_layer': mse_e}, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        lr_e = self.lr_schedulers()[0]
        lr_d = self.lr_schedulers()[1]
        lr_dis = self.lr_schedulers()[2]
        lr_e.step()
        lr_d.step()
        lr_dis.step()

    def validation_step(self, batch, batch_idx):
        real_img = batch['img']
        b_size = real_img.size(0)

        results = self.model(real_img)
        self.curr_device = real_img.device

        nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled = self.model.loss(real_img,
                                                                                             results['recons'],
                                                                                             results['disc_layer_x'],
                                                                                             results['disc_layer_r'],
                                                                                             results['disc_class_x'],
                                                                                             results['disc_class_r'],
                                                                                             results['disc_class_n'],
                                                                                             results['mu'],
                                                                                             results['log_var'])

        loss_encoder = self.kld_weight * kl + mse
        loss_discriminator = bce_dis_original + bce_dis_predicted + bce_dis_sampled
        loss_decoder = self.lambda_mse * mse - (1.0 - self.lambda_mse) * (
                bce_dis_original + bce_dis_predicted + bce_dis_sampled)
        self.log_dict({'val_lossE': loss_encoder, 'val_lossD': loss_decoder, 'val_loss_Dis': loss_discriminator,
                       'val_real_kl': kl, 'val_loss_rec': nle, 'val_loss_layer': mse})

    def test_step(self, batch: Any, batch_idx: int):
        real_img = batch['img']
        b_size = real_img.size(0)

        results = self.model(real_img)
        self.curr_device = real_img.device

        nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled = self.model.loss(real_img,
                                                                                             results['recons'],
                                                                                             results['disc_layer_x'],
                                                                                             results['disc_layer_r'],
                                                                                             results['disc_class_x'],
                                                                                             results['disc_class_r'],
                                                                                             results['disc_class_n'],
                                                                                             results['mu'],
                                                                                             results['log_var'])

        loss_encoder = self.kld_weight * kl + mse
        loss_discriminator = bce_dis_original + bce_dis_predicted + bce_dis_sampled
        loss_decoder = self.lambda_mse * mse - (1.0 - self.lambda_mse) * (
                bce_dis_original + bce_dis_predicted + bce_dis_sampled)

        self.log_dict({'test_lossE': loss_encoder, 'test_lossD': loss_decoder, 'test_loss_Dis': loss_discriminator,
                       'test_real_kl': kl, 'test_loss_rec': nle, 'test_loss_layer': mse})
        return {'recons': results['recons']}

    def sample_images(self, data_set):
        # Get sample reconstruction image
        test_input = data_set['img']
        test_input = test_input.to(self.curr_device)
        recons = self.forward(test_input)['recons']
        test_input=self.denormalize(test_input)
        recons=self.denormalize(recons)
        output = torch.cat((torch.unsqueeze(test_input[0, :, :, :], 0), torch.unsqueeze(recons[0, :, :, :], 0)), 0)
        for i in range(1, recons.shape[0]):
            output = torch.cat((output, torch.unsqueeze(test_input[i, :, :, :], 0)), 0)
            output = torch.cat((output, torch.unsqueeze(recons[i, :, :, :], 0)), 0)

        samples = self.model.sample(120,
                                    self.curr_device,
                                    )
        samples=self.denormalize(samples)
        return output, samples.cpu().data

    def denormalize(self,img):
        std=1/(torch.Tensor([0.1955,0.1824,0.1763])+1e-7)
        mean=-1*torch.Tensor([0.6513,0.4290,0.3931])*std
        denorma_transform=transforms.Normalize(mean=mean, std=std)
        return denorma_transform(img)


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer_e = optim.RMSprop(self.model.encoder.parameters(),
                                    lr=float(self.params['LR']), alpha=0.9, eps=1e-8, weight_decay=0, momentum=0,
                                    centered=False)
        optims.append(optimizer_e)

        optimizer_d = optim.RMSprop(self.model.decoder.parameters(),
                                    lr=float(self.params['LR']), alpha=0.9, eps=1e-8, weight_decay=0, momentum=0,
                                    centered=False)
        optims.append(optimizer_d)

        optimizer_dis = optim.RMSprop(self.model.discriminator.parameters(),
                                      lr=float(self.params['LR']), alpha=0.9, eps=1e-8, weight_decay=0, momentum=0,
                                      centered=False)
        optims.append(optimizer_dis)
        # Check if more than 1 optimizer is required (Used for adversarial training)

        scheduler_e = optim.lr_scheduler.MultiStepLR(optims[0],
                                                     milestones=(200, 350), gamma=0.1)
        # scheduler=optim.lr_scheduler.ReduceLROnPlateau(optims[0])
        scheds.append(scheduler_e)

        scheduler_d = optim.lr_scheduler.MultiStepLR(optims[1],
                                                     milestones=(200, 350), gamma=0.1)
        # scheduler=optim.lr_scheduler.ReduceLROnPlateau(optims[0])
        scheds.append(scheduler_d)

        scheduler_dis = optim.lr_scheduler.MultiStepLR(optims[2],
                                                       milestones=(200, 350), gamma=0.1)
        # scheduler=optim.lr_scheduler.ReduceLROnPlateau(optims[0])
        scheds.append(scheduler_dis)

        return optims, scheds
