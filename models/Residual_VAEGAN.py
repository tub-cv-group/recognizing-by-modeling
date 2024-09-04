import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy
from models import BaseVAE

from .types_ import *


class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False, t=False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = F.relu(ten, False)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = F.relu(ten, True)
            return ten


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
    def __init__(self, cdim=3, z_size=512, channels=(64, 128, 256, 512, 512), image_size=128):
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
        self.var = nn.Linear(num_fc_features, self.zdim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        mu = self.mean(y)
        logvar = self.var(y)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512), image_size=128,
                 conv_input_size=None):
        super(Decoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[-1]
        self.conv_input_size = conv_input_size
        if conv_input_size is None:
            num_fc_features = cc * 4 * 4
        else:
            num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]
        self.fc = nn.Sequential(
            nn.Linear(zdim, num_fc_features),
            nn.ReLU(True),
        )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


class Discriminator(nn.Module):
    def __init__(self, channel_in=3, recon_level=3):
        super(Discriminator, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)))
        self.size = 32
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * 16 * self.size, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, img, mode='REC'):
        if mode == "REC":
            if self.recon_levl == 0:
                return img
            for i, lay in enumerate(self.conv):
                # we take the 9th layer as one of the outputs
                if i == self.recon_levl:
                    img, layer_ten = lay(img, True)
                    # we need the layer representations just for the original and reconstructed,
                    # flatten, because it's a convolutional shape
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    img = lay(img)
        else:
            for i, lay in enumerate(self.conv):
                img = lay(img)

            ten = img.view(len(img), -1)
            ten = self.fc(ten)

            return torch.sigmoid(ten)


class ResidualVaeGan(BaseVAE):
    def __init__(self, in_channels: int = 3,
                 latent_dim: int = 128,
                 recon_level: int = 3,
                 img_size: int = 64,
                 **kwargs):
        super(ResidualVaeGan, self).__init__()
        self.in_channels = in_channels
        # latent space size
        self.z_size = latent_dim
        self.encoder = Encoder(cdim=self.in_channels, z_size=self.z_size, image_size=img_size)
        self.decoder = Decoder(zdim=self.z_size, cdim=self.encoder.cdim, image_size=img_size,
                               conv_input_size=self.encoder.conv_output_size)
        self.discriminator = Discriminator(channel_in=in_channels, recon_level=recon_level)
        # self-defined function to init the parameters
        self.init_parameters()
        self.critrions = nn.BCELoss()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /= numpy.sqrt(3)
                    # nn.init.xavier_normal(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    nn.init.uniform_(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x, gen_size=10):

        mus, log_variances = self.encoder(x)
        z = self.reparameterize(mus, log_variances)
        x_tilde = self.decoder(z)

        z_p = Variable(torch.randn(len(x), self.z_size).cuda(), requires_grad=True)
        x_p = self.decoder(z_p)

        disc_layer_x = self.discriminator(x, "REC")  # discriminator for reconstruction
        disc_layer_r = self.discriminator(x_tilde, "REC")

        disc_class_x = self.discriminator(x, "GAN")
        disc_class_r = self.discriminator(x_tilde, "GAN")
        disc_class_n = self.discriminator(x_p, "GAN")
        # disc_class = self.discriminator(x, x_tilde.detach(), x_p.detach(), "GAN")

        # return x_tilde, disc_class, disc_layer, mus, log_variances
        return {'recons': x_tilde, 'mu': mus, 'log_var': log_variances, 'disc_layer_x': disc_layer_x,
                'disc_layer_r': disc_layer_r, 'disc_class_x': disc_class_x, 'disc_class_r': disc_class_r,
                'disc_class_n': disc_class_n}

    def encode(self, x):
        mus, log_variances = self.encoder(x)
        return mus, log_variances

    def decode(self, z):
        recon = self.decoder(z)
        return recon

    def discriminate(self, img, stage):
        output = self.discriminator(img, stage)
        return output

    def loss(self, x, x_tilde, disc_layer_original, disc_layer_predicted, disc_class_original,
             disc_class_predicted, disc_class_sampled, mus, variances):

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = F.mse_loss(x_tilde, x)

        # kl-divergence
        kl = torch.mean(-0.5 * torch.sum(1 + variances - mus ** 2 - variances.exp(), dim=1), dim=0)

        # mse between intermediate layers
        mse = F.mse_loss(disc_layer_predicted, disc_layer_original)

        bce_dis_original = self.critrions(disc_class_original, torch.ones_like(disc_class_original))
        bce_dis_predicted = self.critrions(disc_class_predicted, torch.zeros_like(disc_class_predicted))
        bce_dis_sampled = self.critrions(disc_class_sampled, torch.zeros_like(disc_class_sampled))

        return nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.z_size)

        z = z.to(current_device)

        samples = self.decoder(z)
        return samples
