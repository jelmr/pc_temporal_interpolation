import open3d
from utils.open3d_util import np_to_open3d_pc
import numpy as np
import torch
import sys
from torchvision.utils import make_grid
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 1 #int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.data_loader):

            target = data.y.to(self.device)
            batch = data.batch.to(self.device)

            pc1_idx = data.graph_id == 0
            pc1 = data.pos[pc1_idx, ...].to(self.device)
            pc1_batch = batch[pc1_idx, ...].to(self.device)

            pc2_idx = data.graph_id == 1, ...
            pc2 = data.pos[pc2_idx].to(self.device)
            pc2_batch = batch[pc2_idx].to(self.device)

            # print(pc1.shape)
            # print(pc2.shape)
            # print(target.shape)
            # print(dir(data))
            # print(target)
            #
            # print("pc1: ", pc1[pc1_batch == 0].mean(dim=0))
            # print("pc1: ", target[pc1_batch == 0].mean(dim=0))
            # print("pc1: ", pc2[pc2_batch == 0].mean(dim=0))
            #
            # pc1_open3d = np_to_open3d_pc(pc1[pc1_batch == 0].data.numpy())
            # pc1_open3d.paint_uniform_color([1,0.706,0])
            # #
            # target_open3d = np_to_open3d_pc(target[pc1_batch == 0].data.numpy())
            # target_open3d.paint_uniform_color([0.706,1, 0])
            # #
            # pc2_open3d = np_to_open3d_pc(pc2[pc2_batch == 0].data.numpy())
            # pc2_open3d.paint_uniform_color([1, 0, 0.706])
            # open3d.draw_geometries([pc1_open3d, target_open3d, pc2_open3d])
            # continue
            #sys.exit(1)



            # print("^^^^^^^^^^^^^^^^^^^^^^^^")
            # print("PC1 TYPE ",type(pc1))
            # print("PC1 TYPE ", pc1.dtype)
            # print("PC2 TYPE ",type(pc2))
            # print("^^^^^^^^^^^^^^^^^^^^^^^^")


            # self.logger.info("--------------------------")
            # self.logger.info(">>> BATCH")
            # self.logger.info(batch.size())
            # self.logger.info(">>> PC1 / PC2")
            # self.logger.info(type(pc1))
            # self.logger.info(pc1.size())
            # self.logger.info(pc2.size())
            # self.logger.info(">>> TARGET")
            # self.logger.info(target.size())
            # self.logger.info(">>> BATCH")
            # self.logger.info(batch.size())
            # self.logger.info("--------------------------")

            self.optimizer.zero_grad()
            #print("Feeding batch...")
            output = self.model(pc1, pc2, pc1_batch, pc2_batch)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{:03d}/{:03d} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    len(self.data_loader),
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                #self.writer.add_image('input', make_grid(datasets.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        # if self.do_validation:
        #     val_log = self._valid_epoch(epoch)
        #     log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():

            for batch_idx, data in enumerate(self.valid_data_loader):

                target = data.y.to(self.device)
                batch = data.batch.to(self.device)

                pc1_idx = data.graph_id == 0
                pc1 = data.pos[pc1_idx, ...].to(self.device)
                pc1_batch = batch[pc1_idx, ...].to(self.device)

                pc2_idx = data.graph_id == 1, ...
                pc2 = data.pos[pc2_idx].to(self.device)
                pc2_batch = batch[pc2_idx].to(self.device)

                output = self.model(pc1, pc2, pc1_batch, pc2_batch)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
