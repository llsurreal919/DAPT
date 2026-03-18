

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def get_ed(x, y):
    ed = torch.norm(x - y, dim=-1).reshape(x.shape[0], 1)
    return ed

def assign_kernel_withoutk(in_feat, kernel, M):
    B, Cin, N0 = in_feat.size()
    in_feat_trans = in_feat.permute(0, 2, 1)
    # point_output = torch.matmul(in_feat_trans, kernel).view(B, N0, M, -1)  # b,n0,m,o1
    out_feat_half1 = torch.matmul(in_feat_trans, kernel[:Cin]).view(B, N0, M, -1)  # b,n0,m,o1
    out_feat_half2 = torch.matmul(in_feat_trans, kernel[Cin:]).view(B, N0, M, -1)  # b,n0,m,o1
    if in_feat.size(1) % 2 != 0:
        out_feat_half_coord = torch.matmul(in_feat_trans[:, :, :3], kernel[Cin: Cin + 3]).view(B, N0, M, -1)  # b,n0,m,o1
    else:
        out_feat_half_coord = torch.zeros_like(out_feat_half2)
    return out_feat_half1 + out_feat_half2, out_feat_half1 + out_feat_half_coord
    # return point_output

class ScoreNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], last_bn=False, temp=1):
        super(ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()
        self.temp = temp

        hidden_unit = list() if hidden_unit is None else copy.deepcopy(hidden_unit)
        hidden_unit.append(out_channel)
        hidden_unit.insert(0, in_channel)

        for i in range(1, len(hidden_unit)):  # from 1st hidden to next hidden to last hidden
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1,
                                                   bias=False if i < len(hidden_unit) - 1 else not last_bn))
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))

    def forward(self, xyz, score_norm='softmax'):
        # xyz : B*3*N*K
        B, _, N, K = xyz.size()
        scores = xyz

        for i, conv in enumerate(self.mlp_convs_hidden):
            if i < len(self.mlp_convs_hidden) - 1:   # 4 = len(self.mlp_convs_hidden)
                scores = F.relu(self.mlp_bns_hidden[i](conv(scores)))  # B*8*N*K
            else:  # if the output layer, no ReLU
                scores = conv(scores)
                if self.last_bn:
                    scores = self.mlp_bns_hidden[i](scores)
        if score_norm == 'softmax':
            scores = F.softmax(scores/self.temp, dim=1)  # + 0.5  # B*m*N*K
        elif score_norm == 'sigmoid':
            scores = torch.sigmoid(scores/self.temp)  # + 0.5  # B*m*N*K
        elif score_norm is None:
            scores = scores
        else:
            raise ValueError('Not Implemented!')

        scores = scores.permute(0, 2, 3, 1)  # B*N*K*m

        return scores



class PAConvCUDA(nn.Module):

    def __init__(self, input_dim, output_dim, scorenet_input_c):
        super(PAConvCUDA, self).__init__()
        
        self.score_input = 'ed7'
        self.score_norm = 'softmax'
        self.temp = 1
        self.init = 'kaiming'
        self.hidden = [8, 16, 16]
        self.m = 8
        self.kernel_input = 'neighbor'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scorenet_input_dim = scorenet_input_c
        
        _init = nn.init.kaiming_normal_
        tensor1 = _init(torch.empty(self.m, input_dim*2, output_dim)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(input_dim*2, self.m * output_dim)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)
       
        self.scorenet = ScoreNet(self.scorenet_input_dim, self.m, hidden_unit=self.hidden, last_bn=False, temp=self.temp)

        

    def forward(self, grouped_xyz):

        r"""
            Parameters
            ----------
            
            grouped_xyz : torch.Tensor
                (B, 3, N1, K) tensor of the descriptors of the the features
            
            Returns
            -------
            out_feat : torch.Tensor
                (B, C, N1) tensor of the new_features descriptors
            new_xyz : torch.Tensor
                (B, N1, 3) tensor of the new features' xyz
            """
       
     
        B, _, N1, K = grouped_xyz.size()
        center_xyz = grouped_xyz[..., :1].repeat(1, 1, 1, K)
        grouped_xyz_diff = grouped_xyz - center_xyz  # [B, 3, N1, K]

        ed = get_ed(center_xyz.permute(0, 2, 3, 1).reshape(B * N1 * K, -1),
                    grouped_xyz.permute(0, 2, 3, 1).reshape(B * N1 * K, -1)).reshape(B, 1, N1, K)       # 计算欧式距离=坐标差值平方和再开根号
        xyz = torch.cat((center_xyz, grouped_xyz_diff, ed), dim=1)  # b, C+C+1, n1, k
        scores = self.scorenet(xyz, score_norm=self.score_norm)  # b,n1,k,m
        
        

        return scores  # b,o1,n1,k



