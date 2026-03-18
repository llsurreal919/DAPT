
import torch
import torch.nn as nn
from torch import einsum
from lib.pointops.functions.pointops import grouping



class CRNet(torch.nn.Module):
    """
    Point Cloud Upsampling via Cascaded Refinement Network 

    Input:
        points: Input points, (B, 3, N_input)
        feat: Input points, (B, C, N_input)
    Output:
        up_point: upsampled results, (B, 3, up_ratio * N_input)
    """
    def __init__(self, up_ratio):
        super(CRNet, self).__init__()
        step_up_rate = up_ratio
        self.upsampling_stage_1 = SubNetwork(up_ratio = step_up_rate)
        self.upsampling_stage_2 = SubNetwork(up_ratio = step_up_rate)
        self.refinement_stage = SubNetwork(up_ratio = 1)

    def forward(self, point_cloud, feat=None, snr=None):
        point_cloud = point_cloud.contiguous() 
        p1_pred, up_feat1 = self.upsampling_stage_1(point_cloud, feat, snr)
        p2_pred, up_feat2 = self.upsampling_stage_2(p1_pred, up_feat1, snr)
        p3_pred, _ = self.refinement_stage(p2_pred, up_feat2, snr) 

        p3_pred = p3_pred.permute(0, 2, 1).contiguous() # (B, 1024, 3)
        p2_pred = p2_pred.permute(0, 2, 1).contiguous() # (B, 1024, 3)
        p1_pred = p1_pred.permute(0, 2, 1).contiguous() # (B, 512, 3)

        if self.training:
            return [p1_pred, p2_pred, p3_pred]
        else:
            return p3_pred




class SubNetwork(nn.Module):
    """
    Upsampling or Refinement Subnetwork

    Input:
        points: Input points, (B, 3, N_input)
        feat: Input points, (B, C, N_input)
    Output:
        up_point: upsampled results, (B, 3, up_ratio * N_input)
    """
    def __init__(self, up_ratio=4):
        super(SubNetwork, self).__init__()
        self.feature_extractor = Transformer_extractor(320, 128)
        self.up_unit = Upsampling_unit(up_ratio=up_ratio)  
        self.regressor = MLP_CONV(in_channel=320, layer_dims=[128, 3])
        
        self.local_att = nn.Sequential(
            nn.Conv1d(320, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(320),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(320, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 320, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(320),
        )

        self.sigmoid = nn.Sigmoid()
        self.deconv = nn.ConvTranspose1d(320, 320, 1, stride=1)
        
    def modulation(self, feat, snr):
        B, _, _ = feat.size()
        feat_snr = feat + snr.view(B, 1, 1)  # (B, C, N128)
        feat_l = self.local_att(feat_snr)
        feat_g = self.global_att(feat_snr)
        feats = feat_l + feat_g
        scale = self.sigmoid(feats)
        
        out_feature = feat * scale
        return out_feature

    def forward(self, points, feat, snr):

        out_feature = self.modulation(feat, snr)
        point_feat = self.feature_extractor(points, out_feature)  # (B, C128, N128)
        
        
        feature = self.deconv(point_feat)
        
        up_feat, duplicated_point = self.up_unit(feature, points) # (B, C, N512)# (B, 3, 512)
        offest = self.regressor(up_feat) # (B, 3, 512)
        up_point = duplicated_point + torch.tanh(offest)

        return up_point, up_feat

class Transformer_extractor(nn.Module):
    """
    Point-wise feature extractor.

    Input:
        points: input points, (B, 3, N_input)
        feat: Input points, (B, C, N_input)
    Output:
        point_feat: ouput feature, (B, dim_feat, N_input)
    """
    def __init__(self, dim_feat, hidden_dim):
        super(Transformer_extractor, self).__init__()
        
        
        # self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, dim_feat])
        self.mlp_2 = MLP_CONV(in_channel=dim_feat * 2, layer_dims=[dim_feat*2, dim_feat])
        self.point_transformer = Transformer(dim_feat, dim=hidden_dim)

    
    
    def forward(self, points, feat):
        global_feature = torch.max(feat, 2, keepdim=True)[0]
        feature_2 = torch.cat([feat, global_feature.repeat((1, 1, feat.size(2)))], 1)  #(320+320,128)
        feature_3 = self.mlp_2(feature_2)
        point_feat = self.point_transformer(feature_3, points)
        return point_feat

class Upsampling_unit(nn.Module):
    """
    Point upsampling unit
    
    Input:
        point_feat: input feature, (B, dim_feat, N_input)
        points: input points, (B, 3, N_input)
    Output:
        up_feat: upsampled feature, (B, dim, up_ratio * N_input)
        duplicated_point: upsampled results, (B, 3, up_ratio * N_input)
    """
    def __init__(self, up_ratio=2):
        super(Upsampling_unit, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=320, layer_dims=[128, 64])
        self.mlp_2 = MLP_Res(in_dim=640, hidden_dim=320, out_dim=320)
        self.deconv_branch = nn.ConvTranspose1d(64, 320, up_ratio, up_ratio, bias=False) 
        self.duplicated_branch = nn.Upsample(scale_factor=up_ratio)

    def forward(self, point_feat, points):
        deconved_feat = self.deconv_branch(self.mlp_1(point_feat))  # (B, 320, 128)->(B, 64, 128)->(B, 320, 512)
        duplicated_feat = self.duplicated_branch(point_feat)
        up_feat = self.mlp_2(torch.cat([deconved_feat, duplicated_feat], 1))
        up_feat = torch.relu(up_feat)
        duplicated_point = self.duplicated_branch(points)
        return up_feat, duplicated_point

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample+pad]
    return idx.int()

def query_knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx.int()

class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)
        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out

class Transformer(nn.Module):
    """
    [Point Transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)

    feed forward of transformer
    Args:
        x: Tensor of features, (B, in_channel, n)
        pos: Tensor of positions, (B, 3, n)

    Returns:
        y: Tensor of features with attention, (B, in_channel, n)
        
    """

    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn_point(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping(pos.contiguous(), idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y+identity
