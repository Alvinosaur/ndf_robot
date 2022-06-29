# Taken from PCN-PyTorch: https://github.com/qinglew/PCN-PyTorch/blob/master/visualization/visualization.py

import os.path as osp
import os
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation
import json

from ndf_robot.training import dataio
from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.model.vnn_object_pursuit import VNNOccNet_Pursuit_OP
from ndf_robot.eval.ndf_alignment import NDFAlignmentCheck
import ndf_robot.training.util as train_util
import ndf_robot.utils.util as util
from ndf_robot.eval.viz_pc import plot_pcd_one_view, export_ply


seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_recon', action='store_true')
    parser.add_argument('--sigma', type=float, default=0.025)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

    # see the demo object descriptions folder for other object models you can try
    mug_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(
    ), 'mug_centered_obj_normalized/1a7ba1f4c892e2da30711cdbdbc73924/models/model_normalized.obj')
    bottle_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(
    ), 'bottle_centered_obj_normalized/1a1c0a8d4bad82169f0594e65f756cf5/models/model_normalized.obj')
    bowl_model1 = osp.join(path_util.get_ndf_demo_obj_descriptions(
    ), 'bowl_centered_obj_normalized/1a0a2715462499fbf9029695a3277412/models/model_normalized.obj')
    bowl_model2 = osp.join(path_util.get_ndf_demo_obj_descriptions(
    ), 'bowl_centered_obj_normalized/519f07b4ecb0b82ed82a7d8f544ae151/models/model_normalized.obj')
    bottle_model2 = osp.join(path_util.get_ndf_demo_obj_descriptions(
    ), 'bottle_centered_obj_normalized/6ebe74793197919e93f2361527e0abe5/models/model_normalized.obj')
    mug_model2 = osp.join(path_util.get_ndf_demo_obj_descriptions(
    ), 'mug_centered_obj_normalized/8b1dca1414ba88cb91986c63a4d7a99a/models/model_normalized.obj')

    model_root = osp.join(path_util.get_ndf_model_weights(),
                          "ndf_vnn/ndf_multi_pretraining/checkpoints/")
    base_model_path = osp.join(model_root, "model_step_6300_multi.pth")
    pursuit_root = osp.join(model_root, "pursuit_checkpoints/")
    images_dir = osp.join(pursuit_root, "eval_images")

    # load base model weights (inculding hypernet)
    train_args = json.load(
        open(osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/ndf_multi_pretraining/args.json')))
    model = VNNOccNet_Pursuit_OP(
        dummy_num_objects=1, obj_feat_dim=train_args["obj_feat_dim"],
        latent_dim=256, sigmoid=True).to(util.DEVICE)
    del model.encoder.obj_feats
    del model.decoder.obj_feats
    model.load_state_dict(torch.load(base_model_path), strict=False)

    # load z bases
    z_dir = osp.join(pursuit_root, "zs")
    num_feats = len(os.listdir(z_dir))
    # z_bases = []  # currently using all z feats, not just bases
    z_feats = []
    for zi in range(num_feats):
        weights = torch.load(
            osp.join(z_dir, f"z_{zi}.pth"), map_location=util.DEVICE)
        z_feats.append((weights["z_enc"], weights["z_dec"]))

    # load coeffs
    # TODO: when using actual bases

    # preload object datasets
    objects = ["bottle", "mug", "bowl"]
    obj_datasets = [dataio.JointOccTrainDataset(128, obj_class=obj_class)
                    for obj_class in objects]
    obj_dataloaders = [torch.utils.data.DataLoader(
        obj_dataset, batch_size=1, shuffle=True, num_workers=0) for obj_dataset in obj_datasets]

    # View point cloud completion by network
    for zi, (enc_feats, dec_feats) in enumerate(z_feats):
        for obj, obj_dataset in zip(objects, obj_datasets):
            # Randomly sample some data
            model_input, gt = next(iter(obj_dataloaders[objects.index(obj)]))
            input_pc = model_input["point_cloud"][0].numpy()
            input_coords = model_input["coords"][0].numpy()
            gt_occ = gt["occ"][0].numpy()
            gt_occ_idxs = np.where(gt_occ > 0)[0]  # -1 or 1
            gt_pc = input_coords[gt_occ_idxs]
            model_input = util.dict_to_gpu(model_input)

            import ipdb
            ipdb.set_trace()
            with torch.no_grad():
                output = model(model_input,
                               enc_feats.unsqueeze(0), dec_feats.unsqueeze(0))
                output_occ = output["occ"][0].cpu().numpy()
                # NOTE: sigmoid already applied
                output_occ_idxs = np.where(output_occ > 0)[0]
                output_pc = input_coords[output_occ_idxs]

            plot_pcd_one_view(os.path.join(images_dir, f"z_{zi}_obj_{obj}.png"), [input_pc, output_pc, gt_pc], [
                              'Input', 'Output', 'GT'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            export_ply(os.path.join(
                images_dir, f"z_{zi}_obj_{obj}.ply"), output_pc)

    exit()

    scale1 = 0.25
    scale2 = 0.4
    mesh1 = trimesh.load(obj_model1, process=False)
    mesh1.apply_scale(scale1)
    # different instance, different scaling
    mesh2 = trimesh.load(obj_model2, process=False)
    mesh2.apply_scale(scale2)
    # mesh2 = trimesh.load(obj_model1, process=False)  # use same object model to debug SE(3) equivariance
    # mesh2.apply_scale(scale1)

    # apply a random initial rotation to the new shape
    quat = np.random.random(4)
    quat = quat / np.linalg.norm(quat)
    rot = np.eye(4)
    rot[:-1, :-1] = Rotation.from_quat(quat).as_matrix()
    mesh2.apply_transform(rot)

    if args.visualize:
        show_mesh1 = mesh1.copy()
        show_mesh2 = mesh2.copy()

        offset = 0.1
        show_mesh1.apply_translation([-1.0 * offset, 0, 0])
        show_mesh2.apply_translation([offset, 0, 0])

        scene = trimesh.Scene()
        scene.add_geometry([show_mesh1, show_mesh2])
        scene.show()

    pcd1 = mesh1.sample(5000)
    pcd2 = mesh2.sample(5000)  # point cloud representing different shape
    # pcd2 = copy.deepcopy(pcd1)  # debug with the exact same point cloud
    # pcd2 = mesh1.sample(5000)  # debug with same shape but different sampled points

    ndf_alignment = NDFAlignmentCheck(
        model, pcd1, pcd2, sigma=args.sigma, trimesh_viz=args.visualize)
    ndf_alignment.sample_pts(
        show_recon=args.show_recon, render_video=args.video)
