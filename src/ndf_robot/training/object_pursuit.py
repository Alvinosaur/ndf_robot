import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import shutil
import json
from collections import defaultdict
import itertools
import functools
import copy

from ndf_robot.training import dataio
import ndf_robot.utils.util as util
import ndf_robot.training.util as train_util
from ndf_robot.training.losses import MemoryLoss
from ndf_robot.model.vnn_object_pursuit import VNNOccNet_Pursuit_OP


def freeze(modules):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def unfreeze(modules):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True


def train_net(model: VNNOccNet_Pursuit_OP,
              train_data,
              val_data,
              log_dir,
              train_hypernet,
              z_dir,
              loss_fn,
              express_threshold,
              batch_size,
              enc_coeffs: torch.Tensor,
              dec_coeffs: torch.Tensor,
              enc_feats: torch.Tensor,
              dec_feats: torch.Tensor,
              max_steps=80,
              val_freq=1,
              lr=0.0004,
              mem_loss_coeff=0.04):
    """

    coefficients should be passed in as tensor([1.0]) if not being trained (ie: learning a new z basis)

    """
    # set logger
    log_file = open(os.path.join(log_dir, "log.txt"), "w")

    # optimize
    optim_param = filter(lambda p: p.requires_grad,
                         itertools.chain(
                             model.encoder.hypernet_weight_block.parameters(),
                             model.decoder.hypernet_weight_block.parameters(), [enc_coeffs, enc_feats, dec_coeffs, dec_feats]))
    optimizer = optim.RMSprop(
        optim_param, lr=lr, weight_decay=1e-7, momentum=0.9)

    # Only use singlenet when training hypernetwork since learning new object basis... couldn't represent using existing bases
    if train_hypernet:
        import ipdb
        ipdb.set_trace()  # TODO: save old hypernet inputs and outputs, use for regularziation
        MemLoss = MemoryLoss(Base_dir=z_dir, device=util.DEVICE)
        mem_coeff = mem_loss_coeff

    # write info
    info_text = f"""Starting training:
        Max steps:      {max_steps}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {log_dir}
        z_dir:           {z_dir}
        loss thresh:     {express_threshold}
    """
    util.write_log(log_file, info_text)

    train_input, train_gt = (
        util.dict_to_gpu(train_data[0]), util.dict_to_gpu(train_data[1])
    )
    val_input, val_gt = (
        util.dict_to_gpu(val_data[0]), util.dict_to_gpu(val_data[1])
    )

    # training process
    min_loss = 1e10
    init_loss = None
    model.train()
    for step in range(max_steps):
        weighted_enc_feat = torch.sum(
            enc_coeffs.view(-1, 1) * enc_feats, dim=0)
        weighted_dec_feat = torch.sum(
            dec_coeffs.view(-1, 1) * dec_feats, dim=0)
        save_dict = {
            'enc_coeffs': enc_coeffs.clone(),
            'dec_coeffs': dec_coeffs.clone(),
            'enc_feats': enc_feats.clone(),
            'dec_feats': dec_feats.clone(),
            'weighted_enc_feat': weighted_enc_feat.clone(),
            'weighted_dec_feat': weighted_dec_feat.clone(),
            'hypernet': copy.deepcopy(model.hypernet_state_dict),
        }

        enc_feat_repeat = weighted_enc_feat.unsqueeze(
            0).repeat(batch_size, 1)
        dec_feat_repeat = weighted_dec_feat.unsqueeze(
            0).repeat(batch_size, 1)

        train_output = model(train_input,
                             obj_feats_enc=enc_feat_repeat,
                             obj_feats_dec=dec_feat_repeat)

        train_loss = loss_fn(train_output, train_gt)['occ']

        # optimize
        optimizer.zero_grad()
        train_loss.backward()
        if train_hypernet:
            MemLoss(model, mem_coeff)

        # NOTE: removed gradient clipping!
        # nn.utils.clip_grad_value_(optim_param, 0.1)
        optimizer.step()

        if step % val_freq == 0:
            # evaluate
            model.eval()
            with torch.no_grad():
                val_output = model(val_input,
                                   obj_feats_enc=enc_feat_repeat,
                                   obj_feats_dec=dec_feat_repeat)
                val_loss = loss_fn(val_output, val_gt)['occ'].item()
            model.train()

        if val_loss < min_loss:
            # save coeffs, weighted enc/dec feats, original enc/dec feats and hypernet
            torch.save(save_dict, os.path.join(log_dir, "weights.pth"))
            min_loss = val_loss

        if init_loss is None:
            init_loss = val_loss

        if min_loss < express_threshold:
            break

    model.eval()
    util.write_log(
        log_file, f"Finished training. Init loss: {init_loss}, min loss: {min_loss}, thresh: {express_threshold}")

    return min_loss < express_threshold, min_loss


def have_seen(model, batch_data, obj_feats_enc, obj_feats_dec, loss_fn, threshold, batch_size):
    """
    Checks each existing basis z to see if it represents
    new object well (low segmentation loss)
    """
    # n_test = int(len(dataset) * test_percent)
    # n_rest = len(dataset) - n_test
    # test_set, _ = random_split(dataset, [n_test, n_rest])

    model_input, gt = batch_data
    model_input = util.dict_to_gpu(model_input)
    gt = util.dict_to_gpu(gt)

    all_loss = []
    feat_loss_pairs = []
    min_loss = 1e10
    best_feat = best_idx = None
    for feat_idx, (enc_feat, dec_feat) in enumerate(zip(obj_feats_enc, obj_feats_dec)):
        with torch.no_grad():
            model_output = model(model_input,
                                 obj_feats_enc=enc_feat.unsqueeze(0).repeat(batch_size, 1), obj_feats_dec=dec_feat.unsqueeze(0).repeat(batch_size, 1))
            loss = loss_fn(model_output, gt, val=True)['occ'].item()
        all_loss.append(loss)
        feat_loss_pairs.append((feat_idx, loss))
        if loss < min_loss:
            min_loss = loss
            best_feat = (enc_feat, dec_feat)
            best_idx = feat_idx

    return loss < threshold, min_loss, best_idx, best_feat, feat_loss_pairs


def save_z(model, enc_feat, dec_feat, log_file, z_dir, z_idx,
           hyper_enc=None, hyper_dec=None):
    if hyper_enc is None:
        hyper_enc, hyper_dec = model.get_hypernet_weights(enc_feat, dec_feat)

    util.write_log(log_file, f"Saving 'z_{z_idx}.pth' to {z_dir}")
    torch.save(
        {
            "z_enc": enc_feat.detach(),
            "z_dec": dec_feat.detach(),
            "weights_enc": hyper_enc.detach(),
            "weights_dec": hyper_dec.detach(),
        },
        os.path.join(z_dir, f"z_{z_idx}.pth"),
    )


def pursuit(model, objects, args, batch_size, log_file, express_threshold, loss_fn):
    """
    Since we only have 3 classes, cannot use entire class dataset at a time.
    Rather, we will randomly sample a batch from a dataset and treat that as a "new object" to consider.

    for i in range(num_pursuit_steps):
        sample random object class
        sample random batch from that class
        perform standard pursuit checks on that batch

    """
    # Create folders
    checkpoint_dir = os.path.join(
        "/".join(args.checkpoint_path.split("/")[:-1]),
        "pursuit_checkpoints")
    z_dir = os.path.join(checkpoint_dir, "zs")
    train_util.cond_mkdir(checkpoint_dir)
    train_util.cond_mkdir(z_dir)

    # freeze backbone, train only the hypernets
    freeze([model])
    unfreeze(model.hypernet)

    # Following the original object pursuit, only keep one of the pretrained obj feat bases, save it to base_dir
    n_objects = len(objects)
    rand_feat_idx = np.random.randint(0, n_objects)
    util.write_log(
        log_file, f"Saving pretrained obj feat of: {objects[rand_feat_idx]}")
    base_feats_enc = [model.encoder.obj_feats[rand_feat_idx]]
    base_feats_dec = [model.decoder.obj_feats[rand_feat_idx]]
    save_z(model, enc_feat=base_feats_enc[0].unsqueeze(0),
           dec_feat=base_feats_dec[0].unsqueeze(0),
           log_file=log_file, z_dir=z_dir, z_idx=0)
    init_hyper_enc, init_hyper_dec = (
        model.get_hypernet_weights(
            base_feats_enc[0].unsqueeze(0),
            base_feats_dec[0].unsqueeze(0)))
    util.write_log(
        log_file, f"Saving initial basis as 'z_0.pth' to {z_dir}")
    torch.save(
        {
            "z_enc": base_feats_enc[0],
            "z_dec": base_feats_dec[0],
            "weights_enc": init_hyper_enc,
            "weights_dec": init_hyper_dec,
        },
        os.path.join(z_dir, f"z_0.pth"),
    )
    # obj_counter != num_bases because some novel objects can be expressed
    # as linear combo of existing bases
    obj_counter = 1  # saved one pretrained obj feat

    del model.encoder.obj_feats
    del model.decoder.obj_feats

    # preload object datasets
    obj_datasets = [
        dataio.JointOccTrainDataset(
            128, depth_aug=args.depth_aug, multiview_aug=args.multiview_aug, obj_class=obj_class) for obj_class in objects]
    obj_dataloaders = [
        DataLoader(obj_datasets[obj_class_gt_idx],
                   batch_size=batch_size, shuffle=True,
                   drop_last=True, num_workers=6)
        for obj_class_gt_idx in range(n_objects)]

    base_info = []
    z_info = []
    save_temp_interval = 5
    num_pursuit_steps = 30
    for pursuit_step in range(num_pursuit_steps):
        num_bases = len(obj_feats_dec)
        # sample random object class
        obj_class_gt_idx = random.randint(0, n_objects - 1)
        obj_class_gt = objects[obj_class_gt_idx]

        # sample random batch from that class
        batch_data = next(iter(obj_dataloaders[obj_class_gt_idx]))

        if save_temp_interval > 0 and pursuit_step % save_temp_interval == 0:
            temp_checkpoint_dir = os.path.join(
                checkpoint_dir, f"checkpoint_round_{pursuit_step}")
            train_util.cond_mkdir(temp_checkpoint_dir)
            torch.save(model.hypernet_state_dict, os.path.join(
                temp_checkpoint_dir, f"hypernet.pth"))
            torch.save({'obj_feats_enc': obj_feats_enc, 'obj_feats_dec': obj_feats_dec}, os.path.join(
                temp_checkpoint_dir, "obj_feats.pth"))
            util.write_log(log_file,
                           f"[checkpoint] pursuit round {pursuit_step} has been saved to {temp_checkpoint_dir}")

        # for each new object, create a new dir
        obj_dir = os.path.join(
            checkpoint_dir, "explored_objects", f"obj_{obj_counter}")
        train_util.cond_mkdir(obj_dir)

        new_obj_info = f"""Starting new object: {obj_class_gt}
            round:               {pursuit_step}
            current base num:    {num_bases}
            object index:        {obj_counter}
            output obj dir:      {obj_dir}
        """
        util.write_log(
            log_file, "\n=============================start new object==============================")
        util.write_log(log_file, new_obj_info)

        # ========================================================================================================
        # check if current object has been seen
        seen, loss, best_idx, best_feat, feat_loss_pairs = have_seen(
            model, batch_data, obj_feats_enc, obj_feats_dec, loss_fn, threshold=express_threshold, batch_size=batch_size)

        util.write_log(
            log_file, f"Obj feat idxs and their losses: {feat_loss_pairs}")
        if seen:
            util.write_log(
                log_file, f"Current object {obj_class_gt} has been seen! corresponding obj_idx: {best_idx}, express loss: {loss}")
            shutil.rmtree(obj_dir)
            util.write_log(
                log_file, "\n===============================end object==================================")
            continue
        else:
            util.write_log(
                log_file, f"Current object {obj_class_gt} is novel, min loss: {loss}, most similiar object: {best_idx}, start object pursuit")

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
            val_data = next(iter(obj_dataloaders[obj_class_gt_idx]))
            expressable, min_loss = train_net(model,
                                              train_data=batch_data,
                                              val_data=val_data,
                                              log_dir=coeff_pursuit_dir,
                                              train_hypernet=False,
                                              z_dir=z_dir,
                                              loss_fn=loss_fn,
                                              express_threshold=express_threshold,
                                              enc_coeffs=new_coeffs_enc_v1,
                                              dec_coeffs=new_coeffs_dec_v1,
                                              enc_feats=torch.stack(
                                                  obj_feats_enc),
                                              dec_feats=torch.stack(
                                                  obj_feats_dec),
                                              batch_size=batch_size,
                                              max_steps=80,
                                              val_freq=1,
                                              lr=0.0004)
            util.write_log(log_file, f"training stop, min loss: {min_loss}")
        # ==========================================================================================================
        # (train as a new base) if not, train this object as a new base
        # the condition to retrain a new base
        import ipdb
        ipdb.set_trace()
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
                obj_feats_enc[0], device=util.DEVICE).unsqueeze(0)
            new_obj_base_dec = torch.rand_like(
                obj_feats_dec[0], device=util.DEVICE).unsqueeze(0)

            # train new hypernets with regularization of not changing output weights given existing bases
            expressable, min_loss = train_net(model,
                                              train_data=batch_data,
                                              val_data=val_data,
                                              log_dir=coeff_pursuit_dir,
                                              train_hypernet=True,
                                              z_dir=z_dir,
                                              loss_fn=loss_fn,
                                              express_threshold=express_threshold,
                                              enc_coeffs=ones_coeff,
                                              dec_coeffs=ones_coeff,
                                              enc_feats=new_obj_base_enc,
                                              dec_feats=new_obj_base_dec,
                                              batch_size=batch_size,
                                              max_steps=80,
                                              val_freq=1,
                                              lr=0.0004)    # type: ignore
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
