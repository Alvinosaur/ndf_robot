import sys
import os
import os.path as osp
import configargparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.training import summaries, losses, training, dataio, config
from ndf_robot.utils import path_util
import ndf_robot.utils.util as util

p = configargparse.ArgumentParser()
p.add_argument('-c', '--config_filepath', required=False,
               is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default=osp.join(
    path_util.get_ndf_model_weights(), 'ndf_vnn'), help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--sidelength', type=int, default=128)

# General training options
p.add_argument('--batch_size', type=int, default=2)
p.add_argument('--lr', type=float, default=1e-4,
               help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=1,  # original: 40001
               help='Number of epochs to train for.')
p.add_argument('--batches_per_validation', type=int, default=10,
               help='Number of batches to run validation')

p.add_argument('--steps_per_val', type=int, default=300,
               help='Number of train steps per val and model save')

p.add_argument('--depth_aug', action='store_true', help='depth_augmentation')
p.add_argument('--multiview_aug', action='store_true',
               help='multiview_augmentation')

p.add_argument('--checkpoint_path', default=None,
               help='Checkpoint to trained model.')
p.add_argument('--dgcnn', action='store_true',
               help='If you want to use a DGCNN encoder instead of pointnet (requires more GPU memory)')
opt = p.parse_args()

# Define the loss
root_path = os.path.join(opt.logging_root, opt.experiment_name)
summary_fn = summaries.occupancy_net
loss_fn = val_loss_fn = losses.occupancy_net

# Define objects
objects = ["bottle", "mug", "bowl"]

model = vnn_occupancy_network.VNNOccNet(latent_dim=256).to(util.DEVICE)
if opt.checkpoint_path is not None:
    model.load_state_dict(torch.load(opt.checkpoint_path))


# Create folders for logging
util.cond_mkdir(root_path)

summaries_dir = os.path.join(root_path, 'summaries')
util.cond_mkdir(summaries_dir)

checkpoints_dir = os.path.join(root_path, 'checkpoints')
util.cond_mkdir(checkpoints_dir)

writer = SummaryWriter(summaries_dir)
log_file = open(os.path.join(root_path, 'output.txt'), "w")

# Run training
base_step = 0
base_epoch = 0
for obj in objects:
    util.write_log(log_file, "Training on object: {}".format(obj))
    train_dataset = dataio.JointOccTrainDataset(
        128, depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=obj)
    val_dataset = dataio.JointOccTrainDataset(
        128, phase='val', depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=obj)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  drop_last=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                drop_last=True, num_workers=4)

    model, base_epoch, base_step = training.train_single_object(
        model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=opt.num_epochs,
        lr=opt.lr, loss_fn=loss_fn, summary_fn=summary_fn,
        clip_grad=False, val_loss_fn=val_loss_fn, base_epoch=base_epoch, object_name=obj, writer=writer, steps_per_val=opt.steps_per_val, checkpoints_dir=checkpoints_dir, base_step=base_step, log_file=log_file)

# Evaluate again on the final model
total_steps = opt.num_epochs * len(train_dataloader)
for obj in objects[0:-1]:
    util.write_log(
        log_file, "Evaluating final model on object: {}".format(obj))
    val_dataset = dataio.JointOccTrainDataset(
        128, phase='val', depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=obj)

    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size,
                                shuffle=False, drop_last=True, num_workers=4)

    val_loss = training.eval_model(model, val_dataloader, val_loss_fn,
                                   batches_per_validation=opt.batches_per_validation)
    util.write_log(
        log_file, "Val loss for object {}: {}".format(obj, val_loss))
