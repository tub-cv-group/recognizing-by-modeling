import os
import math

import torch
from torch import optim
import pytorch_lightning as pl
import torchmetrics
from torch.nn import functional as F
import torch.nn as nn

from models import classifier
from models import *
from models.types_ import *


class vae_ia_classificationEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict, ) -> None:
        super(vae_ia_classificationEXperiment, self).__init__()
        self.model_imgs = vae_models[params['model_params']['backbone_1']['name']](
            **params['model_params']['backbone_1'])
        if params['model_params']['backbone_1']['ckpt']:
            ccc = torch.load(params['model_params']['backbone_1']['ckpt'])['state_dict']
            new_weights = self.model_imgs.state_dict()

            for k in list((new_weights).keys()):
                old_key = 'model_imgs.' + k
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
        final_dim = self.params['final_dim'][self.params['context']]
        print(self.params['context'])
        print(final_dim)
        self.classifer = classifier(final_dim, self.params['num_classes'])
        self.train_acc = torchmetrics.Accuracy(num_classes=self.params['num_classes'], task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.params['num_classes'], task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=self.params['num_classes'], task='multiclass')
        self.train_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        self.test_loss = nn.CrossEntropyLoss()

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        pass

    def training_step(self, batch, batch_idx):
        real_img = batch['img']
        b_size = real_img.size(0)

        labels = batch['labels']
        self.curr_device = real_img.device

        mu, logvar = self.model_imgs.encode(real_img)
        z = self.model_imgs.reparameterize(mu, logvar)

        audios = batch['audio']
        mu_a, logvar_a = self.model_audio.encode(audios)
        z_a = self.model_audio.reparameterize(mu_a, logvar_a).detach()

        mm = torch.zeros(1, b_size, 512).to(self.curr_device)
        loggvar = torch.log(torch.ones(1, b_size, 512)).to(self.curr_device)

        mm = torch.cat((mm, mu.unsqueeze(0)), dim=0)
        loggvar = torch.cat((loggvar, logvar.unsqueeze(0)), dim=0)

        mm = torch.cat((mm, mu_a.unsqueeze(0)), dim=0)
        loggvar = torch.cat((loggvar, logvar_a.unsqueeze(0)), dim=0)

        var = torch.exp(loggvar) + 1e-8
        T = 1. / var
        pd_mu = torch.sum(mm * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)

        #
        if self.params['context'] == 'a':
            input_data = z_a
        elif self.params['context'] == 'i':
            input_data = z
        else:
            #input_data = torch.cat((z, z_a), dim=-1)
            input_data=self.model_imgs.reparameterize(pd_mu,pd_logvar)

        results = self.classifer(input_data.detach())
        loss = self.train_loss(results, labels)
        # output=torch.argmax(nn.functional.softmax(results.detach(), dim=1),dim=1)
        output = torch.argmax(results.detach(), dim=1)
        self.train_acc.update(output, labels)

        acc = self.train_acc.compute()
        self.log_dict({'train_loss': loss, 'train_acc': acc}, prog_bar=True)

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

        mm = torch.zeros(1, b_size, 512).to(self.curr_device)
        loggvar = torch.log(torch.ones(1, b_size, 512)).to(self.curr_device)

        mm = torch.cat((mm, mu.unsqueeze(0)), dim=0)
        loggvar = torch.cat((loggvar, logvar.unsqueeze(0)), dim=0)

        mm = torch.cat((mm, mu_a.unsqueeze(0)), dim=0)
        loggvar = torch.cat((loggvar, logvar_a.unsqueeze(0)), dim=0)

        var = torch.exp(loggvar) + 1e-8
        T = 1. / var
        pd_mu = torch.sum(mm * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)

        #
        if self.params['context'] == 'a':
            input_data = mu_a
        elif self.params['context'] == 'i':
            input_data = mu
        else:
            #input_data = torch.cat((z, z_a), dim=-1)
            input_data = pd_mu

        val_results = self.classifer(input_data.detach())
        val_loss = self.val_loss(val_results, labels)
        # val_output = torch.argmax(nn.functional.softmax(val_results.detach(), dim=1), dim=1)
        val_output = torch.argmax(val_results.detach(), dim=1)
        self.val_acc.update(val_output, labels)

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        v_loss = torch.stack([x['val_loss'] for x in outputs])
        self.log_dict({'val_loss': v_loss.mean(), 'val_acc': self.val_acc.compute()})
        self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        real_img = batch['img']
        b_size = real_img.size(0)

        labels = batch['labels']
        self.curr_device = real_img.device

        mu, logvar = self.model_imgs.encode(real_img)
        z = self.model_imgs.reparameterize(mu, logvar)
        # z = z.view(b_size, t_size, -1)
        # z = torch.mean(z, dim=1)
        re_results = self.model_imgs.decode(mu.detach())

        audios = batch['audio']
        mu_a, logvar_a = self.model_audio.encode(audios)
        z_a = self.model_audio.reparameterize(mu_a, logvar_a)

        mm = torch.zeros(1, b_size, 512).to(self.curr_device)
        loggvar = torch.log(torch.ones(1, b_size, 512)).to(self.curr_device)

        mm = torch.cat((mm, mu.unsqueeze(0)), dim=0)
        loggvar = torch.cat((loggvar, logvar.unsqueeze(0)), dim=0)

        mm = torch.cat((mm, mu_a.unsqueeze(0)), dim=0)
        loggvar = torch.cat((loggvar, logvar_a.unsqueeze(0)), dim=0)

        var = torch.exp(loggvar) + 1e-8
        T = 1. / var
        pd_mu = torch.sum(mm * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)

        #
        if self.params['context'] == 'a':
            input_data = mu_a
        elif self.params['context'] == 'i':
            input_data = mu
        else:
            #input_data = torch.cat((z, z_a), dim=-1)
            input_data = pd_mu

        test_results = self.classifer(input_data.detach())
        test_loss = self.test_loss(test_results, labels)
        # test_output = torch.argmax(nn.functional.softmax(test_results.detach(), dim=1), dim=1)
        test_output = torch.argmax(test_results.detach(), dim=1)
        self.test_acc.update(test_output, labels)

        return {'class': test_output, 'test_loss': test_loss, 'recons': re_results}

    def test_epoch_end(self, outputs):
        t_loss = torch.stack([x['test_loss'] for x in outputs])
        self.log_dict({'test_loss': t_loss.mean(), 'test_acc': self.test_acc.compute()})
        self.test_acc.reset()

    def sample_images(self, data_set):
        test_input = data_set['img']
        test_input = test_input.to(self.curr_device)
        #recons = self.forward(test_input)['recons']
        mu, logvar = self.model_imgs.encode(test_input)
        z = self.model_imgs.reparameterize(mu, logvar)
        recons = self.model_imgs.decode(z)
        output = torch.cat((torch.unsqueeze(test_input[0, :, :, :], 0), torch.unsqueeze(recons[0, :, :, :], 0)), 0)
        for i in range(1, recons.shape[0]):
            output = torch.cat((output, torch.unsqueeze(test_input[i, :, :, :], 0)), 0)
            output = torch.cat((output, torch.unsqueeze(recons[i, :, :, :], 0)), 0)

        samples = self.model_imgs.sample(120,
                                    self.curr_device,
                                    )
    # test_input = next(iter(data_set))['audio']
    # test_input = test_input.to(self.curr_device)
    # # recons = self.forward(test_input)['recons']
    # mu, logvar = self.model_audio.encode(test_input)
    # z = self.model_audio.reparameterize(mu, logvar)
    # recons = self.model_audio.decode(z)
    # output = torch.cat((torch.unsqueeze(test_input[0, :, :, :], 0), torch.unsqueeze(recons[0, :, :, :], 0)), 0)
    # for i in range(1, recons.shape[0]):
    #     output = torch.cat((output, torch.unsqueeze(test_input[i, :, :, :], 0)), 0)
    #     output = torch.cat((output, torch.unsqueeze(recons[i, :, :, :], 0)), 0)
    #
    # samples = self.model_audio.sample(120,
    #                                  self.curr_device,
    #                                  )
        return output, samples.cpu().data

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer_e = optim.Adam(self.classifer.parameters(),
                                 lr=float(self.params['LR']), weight_decay=0.01)
        optims.append(optimizer_e)

        # Check if more than 1 optimizer is required (Used for adversarial training)

        scheduler_e = optim.lr_scheduler.MultiStepLR(optims[0],
                                                     milestones=(300, 600), gamma=0.1)
        # scheduler=optim.lr_scheduler.ReduceLROnPlateau(optims[0])
        scheds.append(scheduler_e)

        return optims, scheds
