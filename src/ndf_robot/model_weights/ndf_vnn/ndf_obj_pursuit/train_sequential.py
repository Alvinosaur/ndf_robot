import sys
import os
import os.path as osp
import configargparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import shutil
import json

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import ndf_robot.model.vnn_object_pursuit as vnn_object_pursuit
from ndf_robot.training import summaries, losses, training, dataio, config
from ndf_robot.utils import path_util
import ndf_robot.utils.util as util
import ndf_robot.training.util as train_util
import ndf_robot.training.object_pursuit as object_pursuit


seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


p = configargparse.ArgumentParser()
p.add_argument('-c', '--config_filepath', required=False,
               is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default=osp.join(
    path_util.get_ndf_model_weights(), 'ndf_vnn'), help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--sidelength', type=int, default=128)

# General training options
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--lr', type=float, default=1e-4,
               help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=1,  # original: 40001
               help='Number of epochs to train for.')
p.add_argument('--batches_per_validation', type=int, default=10,
               help='Number of batches to run validation')

p.add_argument('--steps_per_val', type=int, default=300,
               help='Number of train steps per val and model save')
p.add_argument('--obj_feat_dim', type=int, default=128,
               help='Object-specific feature dim')

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
OBJ_CLASS = "all"
objects = ["bottle", "mug", "bowl"]

# Create folders for logging
train_util.cond_mkdir(root_path)

summaries_dir = os.path.join(root_path, 'summaries')
train_util.cond_mkdir(summaries_dir)

checkpoints_dir = os.path.join(root_path, 'checkpoints')
train_util.cond_mkdir(checkpoints_dir)

writer = SummaryWriter(summaries_dir)
log_file = open(os.path.join(root_path, 'output.txt'), "w")

# Copy files
shutil.copy2("src/ndf_robot/training/train_sequential.py",
             os.path.join(root_path, 'train_sequential.py'))
shutil.copy2("src/ndf_robot/training/training.py",
             os.path.join(root_path, 'training.py'))
shutil.copy2("src/ndf_robot/model/vnn_object_pursuit.py",
             os.path.join(root_path, 'vnn_object_pursuit.py'))
shutil.copy2("src/ndf_robot/model/vnn_occupancy_net_pointnet_dgcnn.py",
             os.path.join(root_path, 'vnn_occupancy_net_pointnet_dgcnn.py'))

# Save arguments
with open(os.path.join(root_path, f"args.json"), "w") as outfile:
    json.dump(vars(opt), outfile, indent=4)

############################## Verify Catastrophic Forgetting ##############################
# # Run training
# model = vnn_occupancy_network.VNNOccNet(latent_dim=256).to(util.DEVICE)
# base_step = 0
# base_epoch = 0
# for obj in objects:
#     util.write_log(log_file, "Training on object: {}".format(obj))
#     train_dataset = dataio.JointOccTrainDataset(
#         128, depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=obj)
#     val_dataset = dataio.JointOccTrainDataset(
#         128, phase='val', depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=obj)

#     train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
#                                   drop_last=True, num_workers=6)
#     val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
#                                 drop_last=True, num_workers=4)

#     model, base_epoch, base_step = training.train(
#         model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=opt.num_epochs,
#         lr=opt.lr, loss_fn=loss_fn, summary_fn=summary_fn,
#         clip_grad=False, val_loss_fn=val_loss_fn, base_epoch=base_epoch, object_name=obj, writer=writer, steps_per_val=opt.steps_per_val, checkpoints_dir=checkpoints_dir, base_step=base_step, log_file=log_file)

# # Evaluate again on the final model
# total_steps = opt.num_epochs * len(train_dataloader)
# for obj in objects[0:-1]:
#     util.write_log(
#         log_file, "Evaluating final model on object: {}".format(obj))
#     val_dataset = dataio.JointOccTrainDataset(
#         128, phase='val', depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=obj)

#     val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size,
#                                 shuffle=False, drop_last=True, num_workers=4)

#     val_loss = training.eval_model(model, val_dataloader, val_loss_fn,
#                                    batches_per_validation=opt.batches_per_validation)
#     util.write_log(
#         log_file, "Val loss for object {}: {}".format(obj, val_loss))


############################## Multi-Object Pretraining ##############################
# pretrain_data_prop = 0.3
# model = vnn_object_pursuit.VNNOccNet_Pretrain_OP(
#     num_objects=len(objects), obj_feat_dim=opt.obj_feat_dim,
#     latent_dim=256).to(util.DEVICE)
# train_dataset = dataio.JointOccTrainDataset(
#     128, depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=OBJ_CLASS, train_prop=pretrain_data_prop)
# val_dataset = dataio.JointOccTrainDataset(
#     128, phase='val', depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=OBJ_CLASS, train_prop=pretrain_data_prop)

# train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
#                               drop_last=True, num_workers=6)
# val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,
#                             drop_last=True, num_workers=4)

# # train
# training.train(model=model, train_dataloader=train_dataloader,
#                val_dataloader=val_dataloader, epochs=opt.num_epochs,
#                lr=opt.lr, loss_fn=loss_fn, summary_fn=summary_fn,
#                clip_grad=False, val_loss_fn=val_loss_fn,
#                object_name='multi', writer=writer, steps_per_val=opt.steps_per_val,
#                checkpoints_dir=checkpoints_dir, log_file=log_file)


############################## Object Pursuit ##############################
opt.checkpoint_path = "src/ndf_robot/model_weights/ndf_vnn/ndf_multi_pretraining/checkpoints/model_step_6300_multi.pth"
pursuit_train_data_prop = 0.8
express_threshold = 0.10  # loss threshold, chosen through empirical observation
batch_size = 10  # small number, realistic for online adaptation
model = vnn_object_pursuit.VNNOccNet_Pursuit_OP(
    dummy_num_objects=len(objects), obj_feat_dim=opt.obj_feat_dim,
    latent_dim=256).to(util.DEVICE)

assert opt.checkpoint_path is not None
model.load_state_dict(torch.load(opt.checkpoint_path))

object_pursuit.pursuit(model, objects, args=opt, batch_size=batch_size,
                       log_file=log_file, express_threshold=express_threshold,
                       loss_fn=loss_fn)
