import torch
import torch.nn as nn
import torch.nn.functional as F
from ndf_robot.model.layers_equi import *
from vnn_occupancy_net_pointnet_dgcnn import maxpool, meanpool, ResnetBlockFC

from object_pursuit.model.coeffnet.hypernet_block import HypernetConvBlock, FCBlock


class VNNOccNet_OP(nn.Module):
    def __init__(self,
                 num_objects,
                 obj_feat_dim,
                 latent_dim,
                 sigmoid=True,
                 return_features=False,
                 acts='all',
                 scaling=10.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.scaling = scaling  # scaling up the point cloud/query points to be larger helps
        self.return_features = return_features

        self.encoder = VNN_ResnetPointnet_OP(
            num_objects=num_objects, obj_feat_dim=obj_feat_dim, c_dim=latent_dim)  # modified resnet-18

        self.decoder = DecoderInner_OP(num_objects=num_objects,
                                       obj_feat_dim=obj_feat_dim, dim=3,
                                       z_dim=latent_dim, c_dim=0,
                                       hidden_size=latent_dim,
                                       leaky=True, sigmoid=sigmoid, return_features=return_features, acts=acts)

    def forward(self, input):
        out_dict = {}

        enc_in = input['point_cloud'] * self.scaling
        query_points = input['coords'] * self.scaling

        z = self.encoder(enc_in)

        if self.return_features:
            out_dict['occ'], out_dict['features'] = self.decoder(
                query_points, z)
        else:
            out_dict['occ'] = self.decoder(query_points, z)

        return out_dict

    def extract_latent(self, input):
        enc_in = input['point_cloud'] * self.scaling
        z = self.encoder(enc_in)
        return z

    def forward_latent(self, z, coords):
        out_dict = {}
        coords = coords * self.scaling
        out_dict['occ'], out_dict['features'] = self.decoder(coords, z)

        return out_dict['features']


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)


class HyperNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_hidden_layers=1):
        """
        No need for VN ReLU because input is z feature which has no rotation
        influence.
        """
        super().__init__()
        self.net = [nn.Linear(in_size, hidden_size, bias=False),
                    nn.ReLU(inplace=True)]
        for i in range(num_hidden_layers):
            self.net += [nn.Linear(hidden_size, hidden_size, bias=False),
                         nn.ReLU(inplace=True)]

        self.net += [nn.Linear(hidden_size, out_size, bias=False)]
        self.net = nn.Sequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, x):
        return self.net(x)


class VNN_ResnetPointnet_OP(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, num_objects, obj_feat_dim, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.obj_feat_dim = obj_feat_dim
        self.c_dim = c_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.meta_output = meta_output

        self.z = nn.Parameter(torch.randn(
            (num_objects, obj_feat_dim)), requires_grad=True)
        self.hypernet_weight_block = HyperNet(
            in_size=obj_feat_dim, hidden_size=None, out_size=hidden_dim * c_dim, num_hidden_layers=1)
        self.hypernet_bias_block = HyperNet(
            in_size=obj_feat_dim, hidden_size=None, out_size=c_dim, num_hidden_layers=1)

        self.conv_pos = VNLinearLeakyReLU(
            3, 128, negative_slope=0.2, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(128, 2 * hidden_dim)
        self.block_0 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        # self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(
            hidden_dim, negative_slope=0.2, share_nonlinearity=False)
        self.pool = meanpool

        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(
                c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(
                c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        elif meta_output == 'equivariant_latent_linear':
            self.vn_inv = VNLinear(c_dim, 3)

    def forward(self, p, z_idxs):
        batch_size = p.size(0)
        p = p.unsqueeze(1).transpose(2, 3)
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())

        # feat: torch.Size([32, 3(feat dim), 3(x y z), 1000, 20(knn)])
        feat = get_graph_feature_cross(p, k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)  # mean across knn (local feature aggr)

        net = self.fc_pos(net)

        net = self.block_0(net)
        # mean across all points (global feature aggr), repeat for each point
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        # final mean across all points, get overall PC feature
        net = self.actvn_c(self.pool(net, dim=-1))

        import ipdb
        ipdb.set_trace()
        w = self.hypernet_weight_block(self.z[z_idxs]).view(
            self.hidden_dim, self.c_dim)
        b = self.hypernet_bias_block(self.z[z_idxs])
        c = F.linear(net, w, b)

        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std
        elif self.meta_output == 'equivariant_latent_linear':
            c_std = self.vn_inv(c)
            return c, c_std

        return c


class DecoderInner_OP(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, num_objects, obj_feat_dim, dim=3, z_dim=128, c_dim=128,
                 hidden_size=128, leaky=False, return_features=False, sigmoid=True, acts='all'):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim  # actually 0

        self.acts = acts
        if self.acts not in ['all', 'inp', 'first_rn', 'inp_first_rn']:
            #self.acts = 'all'
            raise ValueError(
                'Please provide "acts" equal to one of the following: "all", "inp", "first_rn", "inp_first_rn"')

        # Submodules
        if z_dim > 0:
            self.z_in = VNLinear(z_dim, z_dim)
        if c_dim > 0:
            self.c_in = VNLinear(c_dim, c_dim)

        self.fc_in = nn.Linear(z_dim * 2 + c_dim * 2 + 1, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)
        self.return_features = return_features

        # Final classification linear layer, map to 1 class to predict
        # binary occupancy for each of the input query points
        # self.fc_out = nn.Linear(hidden_size, 1)

        self.z = nn.Parameter(torch.randn(
            (num_objects, z_dim)), requires_grad=True)
        self.hypernet_weight_block = HyperNet(
            in_size=obj_feat_dim, hidden_size=None, out_size=hidden_size * 1, num_hidden_layers=1)
        self.hypernet_bias_block = HyperNet(
            in_size=obj_feat_dim, hidden_size=None, out_size=1, num_hidden_layers=1)

        self.sigmoid = sigmoid

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, z_idxs, c=None, **kwargs):
        """
        p: query points (B x N x 3)
        output: occupancy prediction (B x N)
        """
        batch_size, T, D = p.size()
        acts = []
        acts_inp = []
        acts_first_rn = []
        acts_inp_first_rn = []

        if isinstance(c, tuple):
            c, c_meta = c

        net = (p * p).sum(2, keepdim=True)  # sq 2-norm of query points

        if self.z_dim != 0:
            z = z.view(batch_size, -1, D).contiguous()
            # dot-prod btwn each point and z dim over the XYZ dim
            # net_z: torch.Size([B, N, z_dim])
            net_z = torch.einsum('bmi,bni->bmn', p, z)
            z_dir = self.z_in(z)
            z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_z, z_inv], dim=2)

        if self.c_dim != 0:
            c = c.view(batch_size, -1, D).contiguous()
            net_c = torch.einsum('bmi,bni->bmn', p, c)
            c_dir = self.c_in(c)
            c_inv = (c * c_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_c, c_inv], dim=2)

        acts.append(net)
        acts_inp.append(net)
        acts_inp_first_rn.append(net)

        net = self.fc_in(net)
        acts.append(net)
        # acts_inp.append(net)
        # acts_inp_first_rn.append(net)

        net = self.block0(net)
        acts.append(net)
        # acts_inp_first_rn.append(net)
        acts_first_rn.append(net)

        net = self.block1(net)
        acts.append(net)
        net = self.block2(net)
        acts.append(net)
        net = self.block3(net)
        acts.append(net)
        net = self.block4(net)
        last_act = net
        acts.append(net)

        # out = self.fc_out(self.actvn(net))
        net = self.actvn(net)

        import ipdb
        ipdb.set_trace()
        w = self.hypernet_weight_block(self.z[z_idxs]).view(
            self.hidden_dim, self.c_dim)
        b = self.hypernet_bias_block(self.z[z_idxs])
        c = F.linear(net, w, b)

        out = out.squeeze(-1)

        if self.sigmoid:
            out = F.sigmoid(out)

        if self.return_features:
            #acts = torch.cat(acts, dim=-1)
            if self.acts == 'all':
                acts = torch.cat(acts, dim=-1)
            elif self.acts == 'inp':
                acts = torch.cat(acts_inp, dim=-1)
            elif self.acts == 'last':
                acts = last_act
            elif self.acts == 'inp_first_rn':
                acts = torch.cat(acts_inp_first_rn, dim=-1)
            acts = F.normalize(acts, p=2, dim=-1)
            return out, acts
        else:
            return out
