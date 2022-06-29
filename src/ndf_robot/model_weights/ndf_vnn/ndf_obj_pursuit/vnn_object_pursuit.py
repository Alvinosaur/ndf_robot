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


class VNNOccNet_Pursuit_OP(VNNOccNet_Pretrain_OP):
    def __init__(self, dummy_num_objects, obj_feat_dim, latent_dim, sigmoid=True, return_features=False, acts='all', scaling=10):
        super().__init__(dummy_num_objects, obj_feat_dim,
                         latent_dim, sigmoid, return_features, acts, scaling)

        self.encoder = VNN_ResnetPointnet_Pursuit_OP(
            dummy_num_objects=dummy_num_objects,
            obj_feat_dim=obj_feat_dim, c_dim=latent_dim)  # modified resnet-18

        self.decoder = DecoderInner_Pursuit_OP(
            dummy_num_objects=dummy_num_objects,
            obj_feat_dim=obj_feat_dim, dim=3,
            z_dim=latent_dim, c_dim=0,
            hidden_size=latent_dim,
            leaky=True, sigmoid=sigmoid, return_features=return_features, acts=acts)

    @property
    def hypernet_state_dict(self):
        return {
            "encoder_hypernet": self.encoder.hypernet_weight_block.state_dict(),
            "decoder_hypernet": self.decoder.hypernet_weight_block.state_dict()}

    @property
    def hypernet(self):
        return [self.encoder.hypernet_weight_block,
                self.decoder.hypernet_weight_block]

    def load_hypernet(self, hypernet_state_dict):
        self.encoder.hypernet_weight_block.load_state_dict(
            hypernet_state_dict['encoder_hypernet'])
        self.decoder.hypernet_weight_block.load_state_dict(
            hypernet_state_dict['decoder_hypernet'])

    def get_hypernet_weights(self, obj_feats_enc, obj_feats_dec):
        if isinstance(obj_feats_enc, list):
            obj_feats_enc = torch.stack(obj_feats_enc)
        if isinstance(obj_feats_dec, list):
            obj_feats_dec = torch.stack(obj_feats_dec)
        enc_out = self.encoder.hypernet_weight_block(obj_feats_enc)
        dec_out = self.decoder.hypernet_weight_block(obj_feats_dec)
        return enc_out, dec_out

    def forward(self, input, obj_feats_enc, obj_feats_dec):
        out_dict = {}

        enc_in = input['point_cloud'] * self.scaling
        query_points = input['coords'] * self.scaling

        z = self.encoder(enc_in, obj_feats_enc)

        if self.return_features:
            out_dict['occ'], out_dict['features'] = self.decoder(
                query_points, z, obj_feats=obj_feats_dec)
        else:
            out_dict['occ'] = self.decoder(
                query_points, z, obj_feats=obj_feats_dec)

        return out_dict

    def extract_latent(self, input, obj_feats_enc):
        enc_in = input['point_cloud'] * self.scaling
        z = self.encoder(enc_in, obj_feats_enc)
        return z

    def forward_latent(self, z, coords, obj_feats_dec):
        out_dict = {}
        coords = coords * self.scaling
        out_dict['occ'], out_dict['features'] = self.decoder(
            p=coords, z=z, obj_feats=obj_feats_dec)

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

        net = self.forward_backbone(p)

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


class VNN_ResnetPointnet_Pursuit_OP(VNN_ResnetPointnet_Pretrain_OP):
    def __init__(self, dummy_num_objects, obj_feat_dim, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__(dummy_num_objects, obj_feat_dim,
                         c_dim, dim, hidden_dim, k, meta_output)

    def forward(self, p, obj_feats):
        B = p.size(0)
        assert obj_feats.shape[0] == B
        p = p.unsqueeze(1).transpose(2, 3)

        net = self.forward_backbone(p)
        w = self.hypernet_weight_block(obj_feats).view(
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
        B = p.size(0)
        net, activations = self.forward_backbone(p, z, c)

        w = self.hypernet_weight_block(self.obj_feats[obj_idxs]).view(
            B, self.hidden_size, 1)
        out = torch.matmul(net, w)
        out = out.squeeze(-1)

        if self.sigmoid:
            out = F.sigmoid(out)

        if self.return_features:
            (acts, acts_inp, acts_first_rn, acts_inp_first_rn, last_act) = activations
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


class DecoderInner_Pursuit_OP(DecoderInner_Pretrain_OP):
    def __init__(self, dummy_num_objects, obj_feat_dim, dim=3, z_dim=128, c_dim=128,
                 hidden_size=128, leaky=False, return_features=False, sigmoid=True, acts='all'):
        """
        Note: dummy_num_objects is only used to obtain the same architecture as the pretrained model so weights can be loaded. In the forward pass,
        higher level logic determines how many and which obj feats to use. 
        """
        super().__init__(dummy_num_objects, obj_feat_dim, dim, z_dim, c_dim,
                         hidden_size, leaky, return_features, sigmoid, acts)

    def forward(self, p, z, obj_feats, c=None, **kwargs):
        """
        p: query points (B x N x 3)
        output: occupancy prediction (B x N)
        """
        B = p.size(0)
        net, activations = self.forward_backbone(p, z, c)

        w = self.hypernet_weight_block(obj_feats).view(
            B, self.hidden_size, 1)
        out = torch.matmul(net, w)
        out = out.squeeze(-1)

        if self.sigmoid:
            out = F.sigmoid(out)

        if self.return_features:
            (acts, acts_inp, acts_first_rn, acts_inp_first_rn, last_act) = activations
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
