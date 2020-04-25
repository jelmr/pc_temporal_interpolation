import os
import os.path as osp
import sys
import datasets as module_datasets # needs to be imported before torch
import json
import argparse
import torch
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import Logger

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ModelNet



def get_instance(module, name, config, *args, **kwargs):
    return getattr(module, config[name]['type'])(*args, **{**config[name]['args'], **kwargs})


def main(config, resume):
    train_logger = Logger()

    # setup datasets instances

    data_loader = get_instance(module_datasets, 'data_loader', config, training=True)
    valid_data_loader = get_instance(module_datasets, 'data_loader', config, training=False)

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    print(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    # for x in data_loader:
    #     print(x)
    #     sys.exit(1)

    # sys.exit(1)

    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      # datasets=train_loader,
                      # valid_data_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = json.load(handle)
        # setting path to save trained models and log files
        path = os.path.join(config['trainer']['save_dir'], config['name'])

    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)['config']
        
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
