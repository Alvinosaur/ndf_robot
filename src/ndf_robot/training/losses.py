import os
import random
import re
import torch
import torch.nn as nn


def occupancy(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ']
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (
        1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def occupancy_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (
        1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def distance_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()

    dist = torch.abs(model_outputs['occ'] - label * 100).mean()
    loss_dict['dist'] = dist

    return loss_dict


def semantic(model_outputs, ground_truth, val=False):
    loss_dict = {}

    label = ground_truth['occ']
    label = ((label + 1) / 2.).squeeze()

    if val:
        loss_dict['occ'] = torch.zeros(1)
    else:
        loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'].squeeze() + 1e-5) + (
            1 - label) * torch.log(1 - model_outputs['occ'].squeeze() + 1e-5)).mean()

    return loss_dict


class MemoryLoss(nn.Module):
    """The loss function for forgetting prevention"""

    def __init__(self, z_dir, device):
        super(MemoryLoss, self).__init__()
        assert(os.path.isdir(z_dir))
        self.z_dir = z_dir
        self.device = device
        self.file_list = [f for f in os.listdir(
            z_dir) if re.match(r'z_\d+.pth', f)]
        # preload
        self._preload(self.file_list)

    def _preload(self, file_list):
        print("preload from ", file_list)
        self.z_enc = []
        self.z_dec = []
        self.hyper_enc = []
        self.hyper_dec = []
        for file in file_list:
            records = torch.load(os.path.join(self.z_dir, file),
                                 map_location=self.device)
            self.z_enc.append(records['z_enc'])
            self.z_dec.append(records['z_dec'])
            self.hyper_enc.append(records['hyper_enc'])
            self.hyper_dec.append(records['hyper_dec'])

    def _l2_loss(self, pred_hyper_enc, pred_hyper_dec, gt_hyper_enc, gt_hyper_dec, coeff=1.0):
        loss = coeff * (
            torch.norm(pred_hyper_enc - gt_hyper_enc) +
            torch.norm(pred_hyper_dec - gt_hyper_dec)) / 2
        # TODO: backward() in a forward() is not a regular way. We do it in this way to prevent CUDA memory overflow, by releasing the computational graph immediately.
        loss.backward()
        print("mem loss: ", loss.item())

    def forward(self, model, mem_coeff):
        index_list = range(len(self.z_enc))
        if len(index_list) > 10:
            sample_len = int(0.2 * len(index_list))
        else:
            sample_len = len(index_list)
        index_list = random.sample(index_list, sample_len)
        for i in index_list:
            pred_hyper_enc, pred_hyper_dec = model.get_hypernet_weights(
                self.z_enc[i], self.z_dec[i])
            gt_hyper_enc = self.hyper_enc[i]
            gt_hyper_dec = self.hyper_dec[i]
            self._l2_loss(pred_hyper_enc=pred_hyper_enc,
                          pred_hyper_dec=pred_hyper_dec, gt_hyper_enc=gt_hyper_enc,
                          gt_hyper_dec=gt_hyper_dec,
                          coeff=mem_coeff)
