import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops.functions import pointops
from paconv import PAConvCUDA
from libs.cuda_lib.functional import assign_score_withk_halfkernel as assemble_pointnet
import copy


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
            if i < len(self.mlp_convs_hidden) - 1:
                scores = F.relu(self.mlp_bns_hidden[i](conv(scores)))
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



def feat_trans_pointnet(point_input, kernel, m):
    """transforming features using weight matrices"""
    # no feature concat, following PointNet
    B, _, N = point_input.size()  # b, cin, n
    point_output = torch.matmul(point_input.permute(0, 2, 1), kernel).view(B, N, m, -1)  # b,n,m,cout=128
    return point_output

def knn(x, k):
    B, _, N = x.size()
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)

    return idx.int(), pairwise_distance

class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 stride=4,
                 num_neighbors=32,
                 m=8,
                 scorenet_input_dim=7):
        assert stride > 1
        super(Down, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.stride = stride
        self.num_neighbors = num_neighbors
        
        _init = nn.init.kaiming_normal_
        tensor1 = _init(torch.empty(m, in_channels, self.out_channels)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(in_channels, m * self.out_channels)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)
        
        self.paco = PAConvCUDA(input_dim=in_channels, output_dim=self.out_channels, scorenet_input_c=scorenet_input_dim)
        
        self.bn = nn.BatchNorm1d(self.out_channels, momentum=0.1) 
        self.activation = nn.ReLU(inplace=True)
        

        
    def forward(self, xyz, feature):
        p = xyz  # (B, n, 3), (B, c, n)
        x = feature
        M = p.size(1) // self.stride
        p_trans = p.transpose(1, 2) # ->(B, 3, N)

        idx = pointops.furthestsampling(p, M)
        p2 = pointops.gathering(p_trans.contiguous(), idx).transpose(1, 2).contiguous()    # 根据索引 idx 从 p1_trans 中提取对应的点集。->(B, M, 3)
        feat = pointops.gathering(x.contiguous(), idx).contiguous()     # 根据索引 idx 从 feat 中提取对应的点集。  ->(B, 3, M)
        #到此为止p2和feat是一样的东西，只是p2（B,M,3），而feat（B,3,M）

        k_idx, _ = knn(feat, k=16)  # feat中共有M个点，从feat中去找到每个点的k个相邻的点的坐标， k_idx=(B,M,k)
        # k_idx = pointops.knnquery(16, p, p2)
        grouped_f = pointops.grouping(x.contiguous(), k_idx)  # (B, C, M, k=16) 
        scores = self.paco(grouped_f) # (B, M, K=16, m=8)   s
        
        x = feat_trans_pointnet(point_input=feat, kernel=self.weightbank, m=8)  # (B, M, m=8, 128) 

        feats = assemble_pointnet(score=scores, point_input=x, knn_idx=k_idx.long(), aggregate='sum')   # b,128,M
        out_feat = self.bn(feats)
        new_features = self.activation(out_feat)
      
       
        return p2, new_features
    

    

    
class PointTransformerBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 num_neighbors=32):
        super(PointTransformerBlock, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.linear1 = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.transformer = PointTransformerLayer(self.out_channels, num_neighbors=num_neighbors)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.linear2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, xyz, features):
        p = xyz
        x = features

        y = self.relu(self.bn1(self.linear1(x)))
        y = self.relu(self.bn(self.transformer([p, y])))
        y = self.bn2(self.linear2(y))
        y += x
        y = self.relu(y)
        return p, y  
    
class PointTransformerLayer(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 num_neighbors=32):
        super(PointTransformerLayer, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.to_query = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_key = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_value = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_pos_enc = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, self.out_channels, kernel_size=1)
        )
        self.to_attn = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        )
        
        self.key_grouper = pointops.QueryAndGroup(nsample=num_neighbors, return_idx=True)
        self.value_grouper = pointops.QueryAndGroup(nsample=num_neighbors, use_xyz=False)
        self.softmax = nn.Softmax(dim=-1) # (B, C_out, N, K)
        
    def forward(self, px):
        # points, p: (B, N, 3)
        # in_features, x: (B, C_in, N)
        p, x = px
        
        # query, key, and value
        q = self.to_query(x) # (B, C_out, N)
        k = self.to_key(x) # (B, C_out, N)
        v = self.to_value(x) # (B, C_out, N)
        
        # neighbor search
        n_k, _, n_idx = self.key_grouper(xyz=p.contiguous(), features=k.contiguous()) # (B, 3+C_out, N, K)
        n_v, _ = self.value_grouper(xyz=p.contiguous(), features=v.contiguous(), idx=n_idx.int()) # (B, C_out, N, K)
        
        # relative positional encoding
        n_r = self.to_pos_enc(n_k[:, 0:3, :, :]) # (B, C_out, N, K)
        n_v = n_v + n_r
        
        # self-attention
        a = self.to_attn(q.unsqueeze(-1) - n_k[:, 3:, :, :] + n_r) # (B, C_out, N, K)
        a = self.softmax(a)
        y = torch.einsum('b c i j, b c i j -> b c i', a, n_v)
        return y
    
# class ReconstructionLayer(nn.Module):
#     def __init__(self, ratio, output_channel, max_feat):
#         super(ReconstructionLayer, self).__init__()
        
#         self.deconv_features = nn.ConvTranspose1d(in_channels=max_feat, out_channels=output_channel, kernel_size=ratio, stride=ratio)

#     def forward(self, x):
        
#         feature = self.deconv_features(x.permute(0, 2, 1))
#         return feature
   
   
    
#  non_R_adaptive:
     
class ReconstructionLayer(nn.Module):
    def __init__(self, ratio, input_channel, output_channel):
        super(ReconstructionLayer, self).__init__()
        self.deconv_features = nn.ConvTranspose1d(input_channel, output_channel, ratio, stride=ratio)

    def forward(self, x):
        feature = self.deconv_features(x.permute(0, 2, 1))
        return feature

