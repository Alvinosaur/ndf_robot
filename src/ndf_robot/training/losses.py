import os
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

    def __init__(self, Base_dir, device):
        super(MemoryLoss, self).__init__()
        assert(os.path.isdir(Base_dir))
        self.Base_dir = Base_dir
        self.device = device
        self.file_list = [os.path.join(Base_dir, file) for file in os.listdir(
            Base_dir) if file.endswith(".json")]
        # preload
        self._preload(self.file_list)

    def _preload(self, file_list):
        print("preload from ", file_list)
        self.z_enc = []
        self.z_dec = []
        self.weights_enc = []
        self.weights_dec = []
        for file in file_list:
            records = torch.load(file, map_location=self.device)
            self.z_enc.append(records['z_enc'])
            self.z_dec.append(records['z_dec'])
            self.weights_enc.append(records['weights_enc'])
            self.weights_dec.append(records['weights_dec'])

    def _l2_loss(self, pred, gt, coeff=1.0):
        for param in gt:
            loss = coeff * torch.norm(pred[param] - gt[param])
            # TODO: backward() in a forward() is not a regular way. We do it in this way to prevent CUDA memory overflow, by releasing the computational graph immediately.
            loss.backward()

    def forward(self, hypernet, mem_coeff):
        index_list = range(len(self.z))
        if len(index_list) > 10:
            sample_len = int(0.2 * len(index_list))
        else:
            sample_len = len(index_list)
        index_list = random.sample(index_list, sample_len)
        for i in index_list:
            pred_w = hypernet(self.z[i])
            gt_w = self.weights[i]
            self._l2_loss(pred_w, gt_w, mem_coeff)
