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


def freeze(modules):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def unfreeze(modules):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True


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
"""
High-level 

"""
pursuit_train_data_prop = 0.8
express_threshold = 0.16  # chosen through empirical observation
batch_size = 32
model = vnn_object_pursuit.VNNOccNet_Pursuit_OP(
    dummy_num_objects=len(objects), obj_feat_dim=opt.obj_feat_dim,
    latent_dim=256).to(util.DEVICE)


# freeze backbone, train only the hypernets
for name, param in model.named_parameters():
    param.requires_grad = False
unfreeze(model.hypernet)


train_dataset = dataio.JointOccTrainDataset(
    128, depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=OBJ_CLASS, train_prop=pursuit_train_data_prop)
val_dataset = dataio.JointOccTrainDataset(
    128, phase='val', depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=OBJ_CLASS, train_prop=pursuit_train_data_prop)

assert opt.checkpoint_path is not None
model.load_state_dict(torch.load(opt.checkpoint_path))
checkpoint_dir = "/".join(opt.checkpoint_path.split("/")[:-1])

checkpoint_dir = os.path.join(checkpoint_dir, "pursuit_checkpoints")
z_dir = os.path.join(checkpoint_dir, "zs")
train_util.cond_mkdir(checkpoint_dir)
train_util.cond_mkdir(z_dir)


def train_net(z_dim,
              base_num,
              dataset,
              device,
              net_type,
              hypernet,
              backbone=None,
              zs=None,
              save_cp_path=None,
              z_dir=None,
              max_epochs=80,
              batch_size=16,
              lr=0.0004,
              val_percent=0.1,
              wait_epochs=3,
              acc_threshold=1.0,
              l1_loss_coeff=0.2,
              mem_loss_coeff=0.04):
    # set logger
    log_file = open(os.path.join(save_cp_path, "log.txt"), "w")

    # set network
    if net_type == "singlenet":
        primary_net = Singlenet(z_dim)
    elif net_type == "coeffnet":
        assert zs is not None and len(zs) == base_num
        primary_net = Coeffnet(base_num, nn_init=True)
    else:
        raise NotImplementedError

    primary_net.to(device)

    # set dataset and dataloader
    maximum_len = 2500
    if len(dataset) > maximum_len:
        n_data = maximum_len
    else:
        n_data = len(dataset)

    # set train/val dataset
    if val_percent < 1.0:
        n_val = int(n_data * val_percent)
        n_train = int(n_data * (1 - val_percent))
        n_rest = len(dataset) - n_val - n_train
        train, val, _ = random_split(dataset, [n_train, n_val, n_rest])
        train_loader = DataLoader(train, batch_size=batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                                num_workers=8, pin_memory=True, drop_last=True)
    else:
        n_train = n_data
        n_val = n_data
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # optimize
    if backbone is not None:
        optim_param = filter(lambda p: p.requires_grad, itertools.chain(
            primary_net.parameters(), hypernet.parameters(), backbone.parameters()))
    else:
        optim_param = filter(lambda p: p.requires_grad, itertools.chain(
            primary_net.parameters(), hypernet.parameters()))
    optimizer = optim.RMSprop(
        optim_param, lr=lr, weight_decay=1e-7, momentum=0.9)

    # TODO: only add parameters with requires grad = True
    import ipdb
    ipdb.set_trace()
    optim_param = filter(lambda p: p.requires_grad, itertools.chain(
        primary_net.parameters(), hypernet.parameters(), backbone.parameters()))

    # Only use singlenet when training hypernetwork since learning new object basis... couldn't represent using existing bases
    if net_type == "singlenet":
        MemLoss = MemoryLoss(Base_dir=z_dir, device=device)
        mem_coeff = mem_loss_coeff

    global_step = 0
    max_valid_acc = 0
    max_record = None
    stop_counter = 0

    # write info
    info_text = f("""Starting training:
        net type:        {net_type}
        Max epochs:      {max_epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp_path}
        Device:          {device}
        z_dir:           {z_dir}
        wait epochs:     {wait_epochs}
        val acc thres:   {acc_threshold}
        trainable parameter number of the primarynet: {sum(x.numel() for x in primary_net.parameters() if x.requires_grad)}
        trainable parameter number of the hypernet: {sum(x.numel() for x in hypernet.parameters() if x.requires_grad)}
    """)
    write_log(log_file, info_text)

    # training process
    try:
        for epoch in range(max_epochs):
            set_train(primary_net, hypernet, backbone)
            val_list = []
            write_log(log_file, f"Start epoch {epoch}")
            with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{max_epochs}', unit='img") as pbar:
                for batch in train_loader:
                    imgs = batch['image']
                    true_masks = batch['mask']
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(
                        device=device, dtype=torch.float32)

                    if net_type == "singlenet":
                        masks_pred = primary_net(imgs, hypernet, backbone)
                    elif net_type == "coeffnet":
                        masks_pred = primary_net(imgs, zs, hypernet, backbone)
                    else:
                        raise NotImplementedError

                    seg_loss = F.binary_cross_entropy_with_logits(masks_pred, true_masks, pos_weight=torch.tensor([
                                                                  get_pos_weight_from_batch(true_masks)]).to(device))
                    regular_loss = primary_net.L1_loss(l1_loss_coeff)
                    loss = seg_loss + regular_loss
                    pbar.set_postfix(**{'seg loss (batch)': loss.item()})

                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    if net_type == "singlenet":
                        MemLoss(hypernet, mem_coeff)
                        # pass

                    nn.utils.clip_grad_value_(optim_param, 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    global_step += 1

                    # eval
                    if global_step % int(n_train / (batch_size)) == 0:
                        val_score = eval_net(
                            net_type, primary_net, val_loader, device, hypernet, backbone, zs)
                        val_list.append(val_score)
                        write_log(
                            log_file, f"  Validation Dice Coeff: {val_score}, segmentation loss + l1 loss: {loss}")

            if save_cp_path is not None:
                if len(val_list) > 0:
                    avg_valid_acc = sum(val_list) / len(val_list)
                    if avg_valid_acc > max_valid_acc:
                        if net_type == "singlenet":
                            max_record = primary_net.z
                            torch.save(primary_net.state_dict(), os.path.join(
                                save_cp_path, f"Best_z.pth"))
                        elif net_type == "coeffnet":
                            max_record = primary_net.coeffs
                            torch.save(primary_net.state_dict(), os.path.join(
                                save_cp_path, f"Best_coeff.pth"))
                        max_valid_acc = avg_valid_acc
                        stop_counter = 0
                        write_log(
                            log_file, f"epoch {epoch} checkpoint saved! best validation acc: {max_valid_acc}")
                    else:
                        stop_counter += 1

                    if stop_counter >= wait_epochs or max_valid_acc > acc_threshold:
                        # stop procedure
                        write_log(
                            log_file, f"training stopped at epoch {epoch}")
                        write_log(
                            log_file, f"current record value (coeff or z): {max_record}")
                        log_file.close()
                        return max_valid_acc, primary_net
    except Exception as e:
        write_log(log_file, f"Error catch during training! info: {e}")
        return 0.0, primary_net

    # stop procedure
    write_log(log_file, f"training stopped")
    write_log(log_file, f"current record value (coeff or z): {max_record}")
    log_file.close()
    return max_valid_acc, primary_net


def have_seen(dataset, device, z_dir, z_dim, hypernet, backbone, threshold, start_index=0, test_percent=0.2, batch_size=64):
    """
    Checks each existing basis z to see if it represents
    new object well (low segmentation loss)  
    """
    primary_net = Singlenet(z_dim)
    primary_net.to(device)

    n_test = int(len(dataset) * test_percent)
    n_rest = len(dataset) - n_test
    test_set, _ = random_split(dataset, [n_test, n_rest])
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    all_test_acc = []
    z_files = [os.path.join(z_dir, zf) for zf in sorted(
        os.listdir(z_dir)) if zf.endswith('.json')]
    max_acc = 0.0
    max_zf = None
    count = 0
    for zf in z_files:
        if count < start_index:
            count += 1
            continue
        primary_net.load_z(zf)
        test_acc = eval_net(net_type="singlenet", primary_net=primary_net,
                            loader=test_loader, device=device, hypernet=hypernet, backbone=backbone)
        all_test_acc.append(test_acc)
        if test_acc > max_acc:
            max_acc = test_acc
            max_zf = zf
        count += 1

    z_acc_pairs = [(zf, acc) for zf, acc in zip(z_files, all_test_acc)]
    if max_acc > threshold:
        return True, max_acc, max_zf, z_acc_pairs
    else:
        return False, max_acc, max_zf, z_acc_pairs


def pursuit():
    """
    Since we only have 3 classes, cannot use entire class dataset at a time.
    Rather, we will randomly sample a batch from a dataset and treat that as a "new object" to consider.

    for i in range(num_pursuit_steps):
        sample random object class
        sample random batch from that class
        perform standard pursuit checks on that batch

    """
    # Following the original object pursuit, only keep one of the pretrained obj feat bases
    rand_feat_idx = np.random.randint(0, len(objects))
    obj_feats_enc = [model.encoder.obj_feats[rand_feat_idx]]
    obj_feats_dec = [model.decoder.obj_feats[rand_feat_idx]]
    obj_hyper_enc, obj_hyper_dec = (
        model.get_hypernet_weights(obj_feats_enc, obj_feats_dec))
    del model.encoder.obj_feats
    del model.decoder.obj_feats

    # obj_counter != num_bases because some novel objects can be expressed
    # as linear combo of existing bases
    obj_counter = 1  # saved one pretrained obj feat
    base_info = []
    z_info = []
    save_temp_interval = 8
    num_pursuit_steps = 30
    for counter in range(num_pursuit_steps):
        num_bases = len(obj_feats_dec)
        # sample random object class
        obj_class_gt = random.choice(objects)
        # sample random batch from that class
        train_dataset = dataio.JointOccTrainDataset(
            128, depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, obj_class=obj_class_gt)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      drop_last=True, num_workers=6)
        batch_data = next(iter(train_dataloader))

        if save_temp_interval > 0 and counter % save_temp_interval == 0:
            temp_checkpoint_dir = os.path.join(
                checkpoint_dir, f"checkpoint_round_{counter}")
            train_util.cond_mkdir(temp_checkpoint_dir)
            torch.save(model.hypernet_state_dict, os.path.join(
                temp_checkpoint_dir, f"hypernet.pth"))
            torch.save({'obj_feats_enc': obj_feats_enc, 'obj_feats_dec': obj_feats_dec}, os.path.join(
                temp_checkpoint_dir, "obj_feats.pth"))
            util.write_log(log_file,
                           f"[checkpoint] pursuit round {counter} has been saved to {temp_checkpoint_dir}")

        # for each new object, create a new dir
        obj_dir = os.path.join(
            checkpoint_dir, "explored_objects", f"obj_{obj_counter}")
        train_util.cond_mkdir(obj_dir)

        new_obj_info = f"""Starting new object:
            round:               {counter}
            current base num:    {num_bases}
            object index:        {obj_counter}
            output obj dir:      {obj_dir}
        """
        util.write_log(
            log_file, "\n=============================start new object==============================")
        util.write_log(log_file, new_obj_info)

        # ========================================================================================================
        # check if current object has been seen
        seen, loss, obj_idx = have_seen(
            batch_data, obj_feats_enc, obj_feats_dec, express_threshold)

        if seen:
            util.write_log(
                log_file, f"Current object {obj_class_gt} has been seen! corresponding obj_idx: {obj_idx}, express loss: {loss}")
            shutil.rmtree(obj_dir)
            util.write_log(
                log_file, "\n===============================end object==================================")
            continue
        else:
            util.write_log(
                log_file, f"Current object {obj_class_gt} is novel, min loss: {loss}, most similiar object: {obj_idx}, start object pursuit")

        # ========================================================================================================
        # (first check) test if a new object can be expressed by other objects
        if num_bases > 0:
            util.write_log(
                log_file, "start coefficient pursuit (first check):")
            # freeze the hypernet temporarily
            freeze(model.hypernet)

            # rand init new coeffs
            new_coeffs_enc_v1 = torch.randn(
                num_bases, device=util.DEVICE, requires_grad=True)
            new_coeffs_dec_v1 = torch.randn(
                num_bases, device=util.DEVICE, requires_grad=True)
            init_value = 1.0 / np.sqrt(num_bases)
            nn.init.constant_(new_coeffs_enc_v1, init_value)
            nn.init.constant_(new_coeffs_dec_v1, init_value)

            # create save folder
            coeff_pursuit_dir = os.path.join(obj_dir, "coeff_pursuit")
            train_util.cond_mkdir(coeff_pursuit_dir)
            util.write_log(
                log_file, f"coeff pursuit result dir: {coeff_pursuit_dir}")

            # perform coefficient pursuit
            expressable, min_loss = train_net(
                batch_data, model, obj_feats_enc, obj_feats_dec, new_coeffs_enc_v1, new_coeffs_dec_v1, coeff_pursuit_dir, express_threshold)
            util.write_log(log_file, f"training stop, min loss: {min_loss}")
        # ==========================================================================================================
        # (train as a new base) if not, train this object as a new base
        # the condition to retrain a new base
        if not expressable:
            util.write_log(
                log_file, "can't be expressed by bases, start to train as new base:")

            # unfreeze hypernet, temporarily save current one
            unfreeze(model.hypernet)
            temp_hypernet_state_dict = model.hypernet_state_dict

            # create save folder
            base_update_dir = os.path.join(obj_dir, "base_update")
            train_util.cond_mkdir(base_update_dir)
            util.write_log(
                log_file, f"base update result dir: {base_update_dir}")

            ones_coeff = torch.ones(
                1, device=util.DEVICE, requires_grad=False)
            new_obj_base_enc = torch.rand_like(
                obj_feats_enc[0], device=util.DEVICE)
            new_obj_base_dec = torch.rand_like(
                obj_feats_dec[0], device=util.DEVICE)

            # train new hypernets with regularization of not changing output weights given existing bases
            expressable, min_loss = train_net(
                batch_data, model, [new_obj_base_enc], [
                    new_obj_base_dec], ones_coeff, ones_coeff, coeff_pursuit_dir, express_threshold,
                obj_feats_enc, obj_feats_dec, obj_hyper_enc, obj_hyper_dec,
                mem_loss_coeff=0.04)
            util.write_log(
                log_file, f"training stop, min loss: {min_loss}")

            # if the object is invalid, reset hypernet/backbone to prev state
            if min_loss >= express_threshold:
                util.write_log(
                    log_file, f"[Warning] current object {obj_class_gt} is unqualified! The loss {min_loss} should be < {express_threshold}, All records will be removed !")

                # TODO: reset backbone too?
                model.load_hypernet(temp_hypernet_state_dict)
                unfreeze(model.hypernet)

                shutil.rmtree(obj_dir)
                util.write_log(
                    log_file, "\n===============================end object=================================")
                continue

            # ======================================================================================================
            # (second check) check new z can now be approximated (expressed by coeffs) by current bases
            if num_bases > 0:
                util.write_log(
                    log_file, f"start to examine whether the object {obj_counter} can be expressed by bases now (second check):")
                # freeze the hypernet and backbone
                freeze(model.hypernet)

                # rand init new coeffs
                new_coeffs_enc_v2 = torch.randn(
                    num_bases, device=util.DEVICE, requires_grad=True)
                new_coeffs_dec_v2 = torch.randn(
                    num_bases, device=util.DEVICE, requires_grad=True)
                init_value = 1.0 / np.sqrt(num_bases)
                nn.init.constant_(new_coeffs_enc_v2, init_value)
                nn.init.constant_(new_coeffs_dec_v2, init_value)

                # create save folder
                check_express_dir = os.path.join(obj_dir, "check_express")
                train_util.cond_mkdir(check_express_dir)
                util.write_log(
                    log_file, f"check express result dir: {check_express_dir}")

                expressable, min_loss = train_net(
                    batch_data, model, obj_feats_enc, obj_feats_dec, new_coeffs_enc_v2, new_coeffs_dec_v2, coeff_pursuit_dir, express_threshold)
            else:
                expressable = False

            if expressable:
                util.write_log(
                    log_file, f"new z can be expressed by current bases, redundant! min loss: {min_loss}, don't add it to bases")

                # save object's coeffs and output hypernet weights
                linear_combo_obj_feats_enc_v2 = torch.sum(
                    new_coeffs_enc_v2 * torch.stack(obj_feats_enc), dim=0)
                linear_combo_obj_feats_dec_v2 = torch.sum(
                    new_coeffs_dec_v2 * torch.stack(obj_feats_dec), dim=0)
                new_obj_hyper_enc_v2, new_obj_hyper_dec_v2 = model.get_hypernet_weights(
                    obj_feats_enc=linear_combo_obj_feats_enc_v2,
                    obj_feats_dec=linear_combo_obj_feats_dec_v2)
                util.write_log(
                    log_file, f"object {obj_counter} pursuit complete, save object z 'z_{'%04d' % obj_counter}.pth' to {z_dir}")
                torch.save(
                    {
                        "z_enc": linear_combo_obj_feats_enc_v2,
                        "z_dec": linear_combo_obj_feats_dec_v2,
                        "weights_enc": new_obj_hyper_enc_v2,
                        "weights_dec": new_obj_hyper_dec_v2
                    },
                    os.path.join(z_dir, f"z_{'%04d' % obj_counter}.pth"),
                )

            else:
                # save z as a new base
                # NOTE: Since hypernetwork has been updated, shouldn't z_net also be updated again?
                new_obj_hyper_enc_v2, new_obj_hyper_dec_v2 = model.get_hypernet_weights(
                    obj_feats_enc=new_obj_base_enc,
                    obj_feats_dec=new_obj_base_dec)
                util.write_log(
                    log_file, f"new z can't be expressed by current bases, not redundant! express min_loss: {min_loss}, add 'base_{'%04d' % num_bases}.pth' to bases")
                torch.save(
                    {
                        "z_enc": new_obj_base_enc,
                        "z_dec": new_obj_base_dec,
                        "weights_enc": new_obj_hyper_enc_v2,
                        "weights_dec": new_obj_hyper_dec_v2
                    },
                    os.path.join(z_dir, f"base_{'%04d' % obj_counter}.pth"),
                )

                # record base info
                base_info.append({
                    "index": obj_counter,
                    "base_file": f"base_{'%04d' % num_bases}.pth",
                    "z_file": f"z_{'%04d' % obj_counter}.pth"
                })

                # save object's z
                util.write_log(
                    log_file, f"object {obj_counter} pursuit complete, save object z 'z_{'%04d' % obj_counter}.pth' to {z_dir}")
                torch.save(
                    {
                        "z_enc": new_obj_base_enc,
                        "z_dec": new_obj_base_dec,
                        "weights_enc": new_obj_hyper_enc,
                        "weights_dec": new_obj_hyper_dec
                    },
                    os.path.join(z_dir, f"z_{'%04d' % obj_counter}.pth"),
                )
            # ======================================================================================================

        else:
            # save object's z
            util.write_log(
                log_file, f"object {obj_counter} pursuit complete, save object z 'z_{'%04d' % obj_counter}.pth' to {z_dir}")

            # save object's coeffs and output hypernet weights
            linear_combo_obj_feats_enc_v1 = torch.sum(
                new_coeffs_enc_v1 * torch.stack(obj_feats_enc), dim=0)
            linear_combo_obj_feats_dec_v1 = torch.sum(
                new_coeffs_dec_v1 * torch.stack(obj_feats_dec), dim=0)
            new_obj_hyper_enc_v1, new_obj_hyper_dec_v1 = model.get_hypernet_weights(
                obj_feats_enc=linear_combo_obj_feats_enc_v1,
                obj_feats_dec=linear_combo_obj_feats_dec_v1)
            torch.save(
                {
                    "z_enc": linear_combo_obj_feats_enc_v1,
                    "z_dec": linear_combo_obj_feats_dec_v1,
                    "weights_enc": new_obj_hyper_enc_v1,
                    "weights_dec": new_obj_hyper_dec_v1
                },
                os.path.join(z_dir, f"z_{'%04d' % obj_counter}.pth"),
            )

        # record object (z) info
        z_info.append({
            "index": obj_counter,
            "z_file": f"z_{'%04d' % obj_counter}.pth"
        })

        # update (save) info files
        with open(os.path.join(checkpoint_dir, "z_info.json"), "w") as f:
            json.dump(z_info, f)
        with open(os.path.join(checkpoint_dir, "base_info.json"), "w") as f:
            json.dump(base_info, f)

        util.write_log(
            log_file, f"save hypernet and backbone to {checkpoint_dir}, move to next object")
        obj_counter += 1

        # save checkpoint
        torch.save(model.hypernet_state_dict, os.path.join(
            checkpoint_dir, f"hypernet.pth"))
        # if backbone is not None:
        #     torch.save(backbone.state_dict(), os.path.join(
        #         checkpoint_dir, f"backbone.pth"))
        util.write_log(
            log_file, "\n===============================end object=================================")

    log_file.close()


def have_seen():
    # specifically optimize only hypernet or obj feats
    pass
