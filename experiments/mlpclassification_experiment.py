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


class MLP_classificationEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict, ) -> None:
        super(MLP_classificationEXperiment, self).__init__()
        self.model = vae_model
        self.params = params['exp_params']
        self.classifer = classifier(self.model.z_size, self.params['num_classes'])

        for name, parameter in self.model.encoder.named_parameters():
            parameter.requires_grad = False
        for name, parameter in self.model.decoder.named_parameters():
            parameter.requires_grad = False
        for name, parameter in self.model.discriminator.named_parameters():
            parameter.requires_grad = False
        self.train_acc = torchmetrics.Accuracy(num_classes=self.params['num_classes'],task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.params['num_classes'],task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=self.params['num_classes'],task='multiclass')
        self.train_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        self.test_loss = nn.CrossEntropyLoss()

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img = batch['img']
        b_size = real_img.size(0)
        #labels = batch['labels']
        labels = batch['labels']
        self.curr_device = real_img.device

        mu, logvar = self.model.encode(real_img)
        z = self.model.reparameterize(mu, logvar)
        #z=self.feature_norm(z)
        results = self.classifer(z.detach())
        loss = self.train_loss(results, labels)
        output = torch.argmax(nn.functional.softmax(results.detach(), dim=1), dim=1)
        self.train_acc.update(output, labels)

        acc = self.train_acc.compute()
        self.log_dict({'train_loss': loss, 'train_acc': acc}, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        real_img = batch['img']
        b_size = real_img.size(0)
        #labels = batch['labels']
        labels = batch['labels']
        self.curr_device = real_img.device

        mu, logvar = self.model.encode(real_img)
        z = self.model.reparameterize(mu, logvar)
        #z = self.feature_norm(z)
        val_results = self.classifer(z.detach())
        val_loss = self.val_loss(val_results, labels)

        val_output = torch.argmax(nn.functional.softmax(val_results.detach(), dim=1), dim=1)
        self.val_acc.update(val_output, labels)

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        v_loss = torch.stack([x['val_loss'] for x in outputs])
        self.log_dict({'val_loss': v_loss.mean(), 'val_acc': self.val_acc.compute()})
        self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        real_img = batch['img']
        b_size = real_img.size(0)
        #labels = batch['labels']
        labels = batch['labels']
        self.curr_device = real_img.device

        mu, logvar = self.model.encode(real_img)
        z = self.model.reparameterize(mu, logvar)
        re_results = self.model.decode(z)
        #z = self.feature_norm(z)
        test_results = self.classifer(z.detach())
        test_loss = self.test_loss(test_results, labels)
        
        test_output = torch.argmax(nn.functional.softmax(test_results.detach(), dim=1), dim=1)
        self.test_acc.update(test_output, labels)

        return {'class': test_output, 'test_loss': test_loss, 'recons': re_results}

    def test_epoch_end(self, outputs):
        t_loss = torch.stack([x['test_loss'] for x in outputs])
        self.log_dict({'test_loss': t_loss.mean(), 'test_acc': self.test_acc.compute()})
        self.test_acc.reset()

    def sample_images(self, data_set):
        # Get sample reconstruction image
        test_input = data_set['img']
        test_input = test_input.to(self.curr_device)
        recons = self.forward(test_input)['recons']
        #recons=self.denormalize(recons)
        #test_input=self.denormalize(test_input)
        output = torch.cat((torch.unsqueeze(test_input[0, :, :, :], 0), torch.unsqueeze(recons[0, :, :, :], 0)), 0)
        for i in range(1, recons.shape[0]):
            output = torch.cat((output, torch.unsqueeze(test_input[i, :, :, :], 0)), 0)
            output = torch.cat((output, torch.unsqueeze(recons[i, :, :, :], 0)), 0)

        samples = self.model.sample(120,
                                    self.curr_device,
                                    )
        #samples = self.denormalize(samples)
        return output, samples.cpu().data


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer_e = optim.Adam(self.classifer.parameters(),
                                 lr=float(self.params['LR']), weight_decay=0.1)
        optims.append(optimizer_e)

        # Check if more than 1 optimizer is required (Used for adversarial training)

        scheduler_e = optim.lr_scheduler.MultiStepLR(optims[0],
                                                     milestones=(35,70),gamma=0.1)
        # scheduler=optim.lr_scheduler.ReduceLROnPlateau(optims[0])
        scheds.append(scheduler_e)

        return optims, scheds


