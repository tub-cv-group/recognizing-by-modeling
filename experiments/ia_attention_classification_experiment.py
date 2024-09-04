import os
import math

import torch
from torch import optim
import pytorch_lightning as pl
import torchmetrics
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms

from models import classifier
from models import *
from models.types_ import *


class ResidualBlock(nn.Module):

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc != outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        return output


class Encoder(nn.Module):
    def __init__(self, cdim=1, z_size=512, channels=(64, 128, 256, 512, 512), image_size=128, **kwargs):
        super(Encoder, self).__init__()
        self.zdim = z_size
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.conv_output_size = self.calc_conv_output_size()

        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        print("num fc features: ", num_fc_features)
        self.mean = nn.Linear(num_fc_features, self.zdim)
        # self.var = nn.Linear(num_fc_features, self.zdim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        mu = self.mean(y)
        # logvar=self.var(y)
        return mu


class att_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN

    def __init__(self, in_dim):
        super(att_Module, self).__init__()
        self.chanel_in = in_dim

        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        #self.bb=nn.BatchNorm1d(512)
    def forward(self, x, y, t=1):
        m_batchsize, latent_dim = x.size()
        proj_query = self.query(y).view(
            m_batchsize, latent_dim, -1)
        proj_key = self.key(x).view(m_batchsize, -1, latent_dim)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(m_batchsize, -1, latent_dim)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, latent_dim)
        # out=self.bb(out)
        # out = self.gamma*out + x
        out = t*out + x
        return out


class vae_ia_attention_classificationEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict, ) -> None:
        super(vae_ia_attention_classificationEXperiment, self).__init__()
        self.model_imgs = vae_models[params['model_params']['backbone_1']['name']](
            **params['model_params']['backbone_1'])

        if params['model_params']['backbone_1']['ckpt']:
            ccc = torch.load(params['model_params']['backbone_1']['ckpt'])['state_dict']
            new_weights = self.model_imgs.state_dict()

            for k in list((new_weights).keys()):
                old_key='model_imgs.'+k
                if old_key in ccc:
                    new_weights[k] = ccc[old_key]
                else:
                    raise NotImplementedError('please give correct checkpoint')
            self.model_imgs.load_state_dict(new_weights)

        for name, parameter in self.model_imgs.named_parameters():
            parameter.requires_grad = False
        #
        self.model_audio = vae_models[params['model_params']['backbone_2']['name']](
            **params['model_params']['backbone_2'])
        if params['model_params']['backbone_2']['ckpt']:
            ccc = torch.load(params['model_params']['backbone_2']['ckpt'])['state_dict']
            new_weights = self.model_audio.state_dict()
            for k in list((new_weights).keys()):
                old_key = 'model_audio.' + k
                if old_key in ccc:
                    new_weights[k] = ccc[old_key]
                else:

                    raise NotImplementedError('please give correct checkpoint')
            self.model_audio.load_state_dict(new_weights)

        for name, parameter in self.model_audio.named_parameters():
            parameter.requires_grad = False

        self.params = params['exp_params']
        final_dim = self.params['final_dim']
        print(self.params['context'])
        self.kldloss_weight = self.params['kldloss_weight']
        print(self.kldloss_weight)
        self.att_mu = att_Module(final_dim)

        self.att_logvar = att_Module(final_dim)

        self.classifer = classifier(final_dim, self.params['num_classes'])
        if params['model_params']['classifier']['ckpt']:
            ccc = torch.load(params['model_params']['classifier']['ckpt'])['state_dict']
            new_weights = self.classifer.state_dict()
            for k in list((new_weights).keys()):
                old_key = 'classifer.' + k
                if old_key in ccc:
                    new_weights[k] = ccc[old_key]
                else:
                    raise NotImplementedError('please give correct checkpoint')
            self.classifer.load_state_dict(new_weights)
        for name, parameter in self.classifer.named_parameters():
            parameter.requires_grad = True


        self.train_acc = torchmetrics.Accuracy(num_classes=self.params['num_classes'], task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.params['num_classes'], task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=self.params['num_classes'], task='multiclass')
        #self.train_loss = nn.CrossEntropyLoss(reduction='none')
        self.train_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        self.test_loss = nn.CrossEntropyLoss()
        
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        pass
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_img = batch['img']
        b_size = real_img.size(0)

        labels = batch['labels']

        self.curr_device = real_img.device

        mu, logvar = self.model_imgs.encode(real_img)
        z = self.model_imgs.reparameterize(mu, logvar)

        audios = batch['audio']
        mu_a, logvar_a = self.model_audio.encode(audios)
        z_a = self.model_audio.reparameterize(mu_a, logvar_a).detach()

        mu_f = self.att_mu(mu.detach(), mu_a.detach())
        logvar_f = self.att_logvar(logvar.detach(), logvar_a.detach())

        # z_f=self.model_imgs.reparameterize(mu_f, logvar_f)

        std = torch.exp(0.5 * logvar_f)
        eps = torch.randn_like(std)
        z_f = eps * std + mu_f

        results = self.classifer(z_f)

        kld = torch.mean(-0.5 * torch.sum(
            1 + logvar-logvar_f - ((mu - mu_f) ** 2) / logvar_f.exp() - (logvar.exp() / logvar_f.exp()), dim=1), dim=0)
        acc_loss = self.train_loss(results, labels)

        loss = acc_loss + self.kldloss_weight * kld

        output = torch.argmax(nn.functional.softmax(results.detach(), dim=1), dim=1)
        # output = torch.argmax(results.detach(), dim=1)
        self.train_acc.update(output, labels)

        acc = self.train_acc.compute()

        self.log_dict({'train_loss': loss, 'train_acc': acc, 'kld': kld.mean(), 'acc_loss': acc_loss.mean()}, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        real_img = batch['img']
        b_size = real_img.size(0)

        labels = batch['labels']
        self.curr_device = real_img.device

        mu, logvar = self.model_imgs.encode(real_img)
        z = self.model_imgs.reparameterize(mu, logvar)
        # z = z.view(b_size, t_size, -1)
        # z = torch.mean(z, dim=1)

        audios = batch['audio']
        mu_a, logvar_a = self.model_audio.encode(audios)
        z_a = self.model_audio.reparameterize(mu_a, logvar_a)

        mu_f = self.att_mu(mu.detach(), mu_a.detach())
        logvar_f = self.att_logvar(logvar.detach(), logvar_a.detach())

        z_f = self.model_imgs.reparameterize(mu_f, logvar_f)

        val_results = self.classifer(z_f)
        val_acc_loss = self.val_loss(val_results, labels)
        val_kld = torch.mean(-0.5 * torch.sum(
            1 + logvar-logvar_f - ((mu - mu_f) ** 2) / logvar_f.exp() - (logvar.exp() / logvar_f.exp()), dim=1), dim=0)
        val_loss = val_acc_loss + self.kldloss_weight * val_kld

        val_output = torch.argmax(nn.functional.softmax(val_results.detach(), dim=1), dim=1)
        #val_output = torch.argmax(val_results.detach(), dim=1)
        self.val_acc.update(val_output, labels)

        return {'val_loss': val_loss, 'val_kld': val_kld, 'val_acc_loss': val_acc_loss}

    def validation_epoch_end(self, outputs):
        v_loss = torch.stack([x['val_loss'] for x in outputs])
        v_acc_loss = torch.stack([x['val_acc_loss'] for x in outputs])
        v_kld = torch.stack([x['val_kld'] for x in outputs])
        self.log_dict({'val_loss': v_loss.mean(), 'val_acc': self.val_acc.compute(), 'val_acc_loss': v_acc_loss.mean(),
                       'val_kld': v_kld.mean()})
        self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int, t=1):
        real_img = batch['img']
        b_size = real_img.size(0)
        img=real_img
        #img=self.denormalize(real_img)
        labels = batch['labels']
        self.curr_device = real_img.device

        mu, logvar = self.model_imgs.encode(real_img)
        z = self.model_imgs.reparameterize(mu, logvar)
        # z = z.view(b_size, t_size, -1)
        # z = torch.mean(z, dim=1)
        re_results = self.model_imgs.decode(z.detach())
        #re_results=self.denormalize(re_results)
        audios = batch['audio']
        mu_a, logvar_a = self.model_audio.encode(audios)
        z_a = self.model_audio.reparameterize(mu_a, logvar_a)

        mu_f = self.att_mu(mu.detach(), mu_a.detach(),t)
        logvar_f = self.att_logvar(logvar.detach(), logvar_a.detach(),t)

        # z_f = self.model_imgs.reparameterize(mu_f, logvar_f)
        std = torch.exp(0.5 * logvar_f)
        eps = torch.randn_like(std)
        z_f = eps * std + mu_f

        ch_recons = self.model_imgs.decode(z_f.detach())
        #ch_recons=self.denormalize(ch_recons)
        test_results = self.classifer(z_f)
        test_acc_loss = self.test_loss(test_results, labels)
        test_kld = torch.mean(-0.5 * torch.sum(
            1 + logvar-logvar_f - ((mu - mu_f) ** 2) / logvar_f.exp() - (logvar.exp() / logvar_f.exp()), dim=1), dim=0)

        test_loss = test_acc_loss + self.kldloss_weight * test_kld

        test_output = torch.argmax(nn.functional.softmax(test_results.detach(), dim=1), dim=1)
        #test_output = torch.argmax(test_results.detach(), dim=1)
        self.test_acc.update(test_output, labels)
        return {'class': test_output, 'test_loss': test_loss, 'recons': re_results, 'c_recons': ch_recons,
                'test_acc_loss': test_acc_loss, 'test_kld': test_kld, 'ori_feature': z.detach(),
                'later_feature': z_f.detach(),'img':img}

    def test_epoch_end(self, outputs):
        t_loss = torch.stack([x['test_loss'] for x in outputs])
        t_acc_loss = torch.stack([x['test_acc_loss'] for x in outputs])
        t_kld = torch.stack([x['test_kld'] for x in outputs])
        self.log_dict({'test_loss': t_loss.mean(), 'test_acc': self.test_acc.compute(), 'test_kld': t_kld.mean(),
                       'test_acc_loss': t_acc_loss.mean()})
        self.test_acc.reset()


    def log_images(self, data_set):
        test_image = data_set['img'].to(self.curr_device)
        # recons = self.forward(test_input)['recons']
        mu, logvar = self.model_imgs.encode(test_image)
        z = self.model_imgs.reparameterize(mu, logvar)

        test_audio = data_set['audio'].to(self.curr_device)
        # recons = self.forward(test_input)['recons']
        mu_a, logvar_a = self.model_audio.encode(test_audio)

        mu_f = self.att_mu(mu.detach(), mu_a.detach())
        logvar_f = self.att_logvar(logvar.detach(), logvar_a.detach())

        z_f = self.model_imgs.reparameterize(mu_f, logvar_f)

        recons = self.model_imgs.decode(z)
        ch_recons = self.model_imgs.decode(z_f)

        output = torch.cat((torch.unsqueeze(test_image[0, :, :, :], 0), torch.unsqueeze(recons[0, :, :, :], 0),
                            torch.unsqueeze(ch_recons[0, :, :, :], 0)), 0)
        for i in range(1, recons.shape[0]):
            output = torch.cat((output, torch.unsqueeze(test_image[i, :, :, :], 0)), 0)
            output = torch.cat((output, torch.unsqueeze(recons[i, :, :, :], 0)), 0)
            output = torch.cat((output, torch.unsqueeze(ch_recons[i, :, :, :], 0)), 0)

        samples = self.model_imgs.sample(120,
                                         self.curr_device,
                                         )
        return output, samples.cpu().data


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer_a = optim.Adam(self.att_mu.parameters(),
                                 lr=float(self.params['LR']))
        optims.append(optimizer_a)

        optimizer_b = optim.Adam(self.att_logvar.parameters(),
                                 lr=float(self.params['LR']))
        optims.append(optimizer_b)
        scheduler_a = optim.lr_scheduler.MultiStepLR(optims[0],
                                                     milestones=(100,), gamma=0.1)
        scheds.append(scheduler_a)
        scheduler_b=optim.lr_scheduler.MultiStepLR(optims[1],
                                                     milestones=(100,), gamma=0.1)
        scheds.append(scheduler_b)

        optimizer_c = optim.Adam(self.classifer.parameters(),
                                 lr=float(self.params['LR']),weight_decay=0.1)
        optims.append(optimizer_c)

        # Check if more than 1 optimizer is required (Used for adversarial training)

        scheduler_c = optim.lr_scheduler.MultiStepLR(optims[2],
                                                     milestones=(70,), gamma=0.1)
        # scheduler=optim.lr_scheduler.ReduceLROnPlateau(optims[0])
        scheds.append(scheduler_c)

        return optims, scheds
