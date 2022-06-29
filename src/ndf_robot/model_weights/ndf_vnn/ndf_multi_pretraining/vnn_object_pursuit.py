import torch
import torch.nn as nn
import torch.nn.functional as F
from ndf_robot.model.layers_equi import *
from ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn import maxpool, meanpool, ResnetBlockFC, \
    VNNOccNet, VNN_ResnetPointnet, DecoderInner


class VNNOccNet_Pretrain_OP(VNNOccNet):
    def __init__(self,
                 num_objects,
                 obj_feat_dim,
                 latent_dim,
                 sigmoid=True,
                 return_features=False,
                 acts='all',
                 scaling=10.0):
        super().__init__(obj_feat_dim, latent_dim, sigmoid, return_features, acts)

        self.encoder = VNN_ResnetPointnet_Pretrain_OP(
            num_objects=num_objects, obj_feat_dim=obj_feat_dim, c_dim=latent_dim)  # modified resnet-18

        self.decoder = DecoderInner_Pretrain_OP(num_objects=num_objects,
                                                obj_feat_dim=obj_feat_dim, dim=3,
                                                z_dim=latent_dim, c_dim=0,
                                                hidden_size=latent_dim,
                                                leaky=True, sigmoid=sigmoid, return_features=return_features, acts=acts)

    def forward(self, input):
        out_dict = {}

        enc_in = input['point_cloud'] * self.scaling
        query_points = input['coords'] * self.scaling
        obj_idxs = input['obj_idxs']

        z = self.encoder(enc_in, obj_idxs)

        if self.return_features:
            out_dict['occ'], out_dict['features'] = self.decoder(
                query_points, z, obj_idxs=obj_idxs)
        else:
            out_dict['occ'] = self.decoder(query_points, z, obj_idxs=obj_idxs)

        return out_dict

    def extract_latent(self, input):
        enc_in = input['point_cloud'] * self.scaling
        obj_idxs = input['obj_idxs']
        z = self.encoder(enc_in, obj_idxs)
        return z

    def forward_latent(self, z, coords, obj_idxs):
        out_dict = {}
        coords = coords * self.scaling
        out_dict['occ'], out_dict['features'] = self.decoder(
            p=coords, z=z, obj_idxs=obj_idxs)

        return out_dict['features']


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
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


def apply_linear_VN_batch(x, w, b=None):
    """
    Apply linear transformation with VN activation function.
    """
    x = x.transpose(1, -1)
    try:
        x = torch.matmul(x, w)
    except:
        import ipdb
        ipdb.set_trace()
    if b is not None:
        x = x + b
    return x.transpose(1, -1)


class VNN_ResnetPointnet_Pretrain_OP(VNN_ResnetPointnet):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, num_objects, obj_feat_dim, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__(c_dim=c_dim, dim=dim, hidden_dim=hidden_dim,
                         k=k, meta_output=meta_output)
        del self.fc_c
        self.hidden_dim = hidden_dim
        self.obj_feat_dim = obj_feat_dim
        self.obj_feats = nn.Parameter(torch.randn((num_objects, obj_feat_dim)),
                                      requires_grad=True)

        self.hypernet_weight_block = HyperNet(
            in_size=obj_feat_dim, hidden_size=obj_feat_dim, out_size=hidden_dim * c_dim, num_hidden_layers=1)

    def forward(self, p, obj_idxs):
        B = p.size(0)
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

        w = self.hypernet_weight_block(self.obj_feats[obj_idxs]).view(
            B, self.hidden_dim, self.c_dim)
        c = apply_linear_VN_batch(net, w)

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


class DecoderInner_Pretrain_OP(DecoderInner):
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
        super().__init__(dim=dim, z_dim=z_dim, c_dim=c_dim,
                         hidden_size=hidden_size, leaky=leaky, return_features=return_features, sigmoid=sigmoid, acts=acts)
        del self.fc_out
        self.hidden_size = hidden_size
        self.num_objects = num_objects
        self.obj_feat_dim = obj_feat_dim
        self.obj_feats = nn.Parameter(torch.randn((num_objects, obj_feat_dim)),
                                      requires_grad=True)

        self.hypernet_weight_block = HyperNet(
            in_size=obj_feat_dim, hidden_size=obj_feat_dim, out_size=hidden_size * 1, num_hidden_layers=1)

        self.sigmoid = sigmoid

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, obj_idxs, c=None, **kwargs):
        """
        p: query points (B x N x 3)
        output: occupancy prediction (B x N)
        """
        B, T, D = p.size()
        acts = []
        acts_inp = []
        acts_first_rn = []
        acts_inp_first_rn = []

        if isinstance(c, tuple):
            c, c_meta = c

        net = (p * p).sum(2, keepdim=True)  # sq 2-norm of query points

        if self.z_dim != 0:
            z = z.view(B, -1, D).contiguous()
            # dot-prod btwn each point and z dim over the XYZ dim
            # net_z: torch.Size([B, N, z_dim])
            net_z = torch.einsum('bmi,bni->bmn', p, z)
            z_dir = self.z_in(z)
            z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_z, z_inv], dim=2)

        if self.c_dim != 0:
            c = c.view(B, -1, D).contiguous()
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

        w = self.hypernet_weight_block(self.obj_feats[obj_idxs]).view(
            B, self.hidden_size, 1)
        out = torch.matmul(net, w)

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
