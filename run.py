import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch
from models import *
from experiments import MLP_classificationEXperiment, \
    ResidualVaeGanEXperiment, \
    vae_ia_classificationEXperiment, vae_ia_attention_classificationEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset, RADFDataset, RADVIDEODataset, CELEBADataset
from callbacks import ModelCheckpointWithArtifactLogging
from callbacks import SaveAndLogConfigCallback
import pytorch_lightning.plugins
from evaluation import norm_evaluation, visu_allimages, interpolation_images, custom_eval, fusion_matrix_evaluation, \
    cal_KL, tsne_eval, one_call
from pytorch_lightning.plugins.io import TorchCheckpointIO
import subprocess

vae_experments = {
    'MLP_class': MLP_classificationEXperiment,
    'Residual_VAEGAN': ResidualVaeGanEXperiment,
    'vae_ia': vae_ia_classificationEXperiment,
    'vae_ia_att': vae_ia_attention_classificationEXperiment,
}

Dataset = {
    'sortedRaFD': RADFDataset,
    'RaFD_cropped': RADFDataset,
    'RAVDESS_CROPP_VA': RADVIDEODataset,
    'cropped_celeba': CELEBADataset,
    'RAVDESS_cropp':CELEBADataset,
    'mead': CELEBADataset
}


def tra(runner, experiments, data, imgs):
    ckpt_path = config['model_params']['ckpt']
    if ckpt_path:
        print('=========loading model==========')
        if not hasattr(experiments, 'classifer'):
            ccc = torch.load(ckpt_path)
            experiments.load_state_dict(ccc['state_dict'])
        else:
            new_weights = experiments.state_dict()
            old_weights = torch.load(ckpt_path)['state_dict']
            for k in list((old_weights).items()):
                new_weights[k[0]] = k[1]
            experiments.load_state_dict(new_weights)
    print(f"======= Training {config['model_params']['model_name']} =======")
    runner.fit(experiments, datamodule=data)


def evv(runner, experiments, data, eval_config):
    ckpt_path = config['model_params']['ckpt']

    if ckpt_path:
        print('=========loading model==========')
        ccc = torch.load(ckpt_path)
        experiments.load_state_dict(ccc['state_dict'])
        norm_evaluation(experiments, data, runner)
    else:
        raise NotImplementedError("No Pretrained model")
        

def inter(runner, experiments, data, eval_config):
    ckpt_path = config['model_params']['ckpt']

    if ckpt_path:
        print('=========loading model==========')
        ccc = torch.load(ckpt_path)
        experiments.load_state_dict(ccc['state_dict'])
        interpolation_images(experiments, data, runner, eval_config['inter'])
    else:
        raise NotImplementedError("No Pretrained model")


def cus(runner, experiments, data, eval_config):
    ckpt_path = config['model_params']['ckpt']

    if ckpt_path:
        print('=========loading model==========')
        ccc = torch.load(ckpt_path)
        experiments.load_state_dict(ccc['state_dict'])
        custom_eval(experiments, data, runner, eval_config['cus'])
    else:
        raise NotImplementedError("No Pretrained model")


def fus(runner, experiments, data, eval_config):
    ckpt_path = config['model_params']['ckpt']

    if ckpt_path:
        print('=========loading model==========')
        ccc = torch.load(ckpt_path)
        experiments.load_state_dict(ccc['state_dict'])
        fusion_matrix_evaluation(experiments, data, runner)
    else:
        raise NotImplementedError("No Pretrained model")


def ckl(runner, experiments, data, eval_config):
    ckpt_path = config['model_params']['ckpt']
    device=torch.device('cuda:1')
    if ckpt_path:
        print('=========loading model==========')
        ccc = torch.load(ckpt_path,map_location=device)
        experiments.load_state_dict(ccc['state_dict'])
        cal_KL(experiments, data, runner, eval_config['ckl'])
    else:
        raise NotImplementedError("No Pretrained model")


def vis_feature(runner, experiments, data, eval_config):
    ckpt_path = config['model_params']['ckpt']

    if ckpt_path:
        print('=========loading model==========')
        ccc = torch.load(ckpt_path)
        experiments.load_state_dict(ccc['state_dict'])
        tsne_eval(experiments, data, runner, eval_config['tsne'])
    else:
        raise NotImplementedError("No Pretrained model")


def one_case(runner, experiments, data, eval_config):
    ckpt_path = config['model_params']['ckpt']

    if ckpt_path:
        print('=========loading model==========')
        ccc = torch.load(ckpt_path)
        experiments.load_state_dict(ccc['state_dict'])
        one_call(experiments, data, runner, args.image, args.audio)
    else:
        raise NotImplementedError("No Pretrained model")


available_commands = {
    'fit': tra,
    'eval': evv,
    'inter': inter,
    'cus': cus,
    'fus': fus,
    'ckl': ckl,
    'tsne': vis_feature,
    'onecall': one_case,
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default=None)
parser.add_argument('--command',
                    dest='command',
                    help='command',
                    default='fit')

parser.add_argument('--eval',
                    dest='eval',
                    help='eval_path',
                    default='configs/eval/inter.yaml')

parser.add_argument('--image',
                    dest='image',
                    help='image_path',
                    default=None)

parser.add_argument('--audio',
                    dest='audio',
                    help='audio_path',
                    default=None)

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
with open(args.eval, 'r') as file:
    try:
        eval_config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['model_name'] + '-' +
                                   config['data_params']['data_path'].split('/')[-1], )
seed_everything(config['exp_params']['manual_seed'], True)

print(config['model_params']['model_name'])
if config['model_params']['backbone_name'] in vae_models:
    model = vae_models[config['model_params']['backbone_name']](**config['model_params'])
else:
    model=None

experiments = vae_experments[config['model_params']['model_name']](model,
                                                                   config)
#

data = Dataset[config['data_params']['data_path'].split('/')[-1]](**config["data_params"],
                                                                  pin_memory=len(config['trainer_params']['gpus']) != 0)

if args.command == 'fit':
    strat = pytorch_lightning.strategies.ddp.DDPStrategy(
        find_unused_parameters=config['exp_params']['find_unused_parameters'])
else:
    strat = None

runner = Trainer(logger=[tb_logger],
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpointWithArtifactLogging(
                         save_top_k=5,
                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                         monitor=config['exp_params']['monitor'],
                         save_last=True,
                         mode=config['exp_params']['mode']
                     ),
                     SaveAndLogConfigCallback(),
                 ],

                 strategy=strat,
                 **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
subprocess.run(f"cp {args.filename} {os.path.join(tb_logger.log_dir, 'hypermeters.yaml')}", shell=True)

available_commands[args.command](runner, experiments, data, eval_config)
