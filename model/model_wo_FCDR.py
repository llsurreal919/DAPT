import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
from pytorch3d.loss import chamfer_distance
from paconv_util import Down, ReconstructionLayer, PointTransformerBlock

from model.pucrn_adaptive import CRNet

def define_CA(in_channel):
    net = None
    net = CA(in_planes=in_channel)
    return net

class CA(nn.Module):
    def __init__(self, in_planes, ratio=8):
        """
        第一层全连接层神经元个数较少, 因此需要一个比例系数ratio进行缩放
        """
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
 
        """
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        """
        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid() 
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class Model(nn.Module):
    def __init__(self, normal_channel=False, bottleneck_size=200, recon_points=2048, channel_name='AWGN',
                 SNR_MIN=0, SNR_MAX=10, SNR_val=5):
        super(Model, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.normal_channel = normal_channel
        self.recon_points = recon_points
    
        self.SNR_MAX = SNR_MAX
        self.SNR_MIN = SNR_MIN
        self.test_SNR = SNR_val

        self.channel_name = channel_name
        self.down1 = Down(in_channels=3, out_channels=128, stride=4, num_neighbors=16, scorenet_input_dim=7)
        self.down2 = Down(in_channels=128, out_channels=bottleneck_size, stride=4, num_neighbors=16, scorenet_input_dim=257)
        
        self.block0 = PointTransformerBlock(in_channels=128, num_neighbors=16)
        self.block1 = PointTransformerBlock(in_channels=bottleneck_size, num_neighbors=16)

        self.CAnet = define_CA(in_channel=320)
        
        self.decompression = ReconstructionLayer(recon_points // 16, bottleneck_size, 128)
        self.coor_reconstruction_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

 
        self.net = CRNet(4)

       
        self.register_buffer('z_mean', torch.tensor(0))
        self.register_buffer('z_std', torch.tensor(1))
        
    def power_norm(self, feature):
        self.z_mean = feature.mean(-1)
        self.z_std = feature.std(-1)
        z_in = (feature-self.z_mean.unsqueeze(-1))/self.z_std.unsqueeze(-1)
        return z_in
    
    def channel(self, data, band):
        if band == self.bottleneck_size:
            data_received = data
        else:
            data_received = data[:, :band]
        data_received = self.power_norm(data_received)
        
        with torch.no_grad():
            noise_std = 10 ** (-self.snr * 1.0 / 20)
            if (self.channel_name == 'Rayleigh'):
                h_numpy = np.random.rayleigh(scale = 1, size = data_received.shape)
                h = torch.FloatTensor(h_numpy).to(data_received.device)
            else:
                h = 1
            AWGN = (noise_std * torch.randn_like(data_received)).to(data_received.device)  # (32, bottleneck_size)
        data_r = h * data_received + AWGN
        return data_r
    
    def calculate_loss(self, recoord, pc_gd, gt_downsample=None, weight_w = torch.tensor(1.0), training=False):
        if (training is True):
            P1, P2, P3  = recoord
            cd_1 = chamfer_distance(P1, gt_downsample)[0]
            cd_2 = chamfer_distance(P2, pc_gd)[0]
            cd_3 = chamfer_distance(P3, pc_gd)[0]
            loss = (cd_1 + cd_2 + cd_3) * (weight_w.to(cd_1.device))
        else:
            loss = chamfer_distance(recoord, pc_gd)[0]
            
        return loss

    def forward(self, xyz, band, training = True, simu_name = 'single_coderate_multipe_snr'):
        B, N, _ = xyz.shape
        if training is True:
            if simu_name=='single_coderate_multipe_snr':
                self.snr =  (torch.rand(B, 1).to(xyz.device)) * (self.SNR_MAX-self.SNR_MIN) + self.SNR_MIN
            elif simu_name=='single_coderate_single_snr':
                self.snr= (torch.ones(B, 1).to(xyz.device)) * self.test_SNR
        else:
            self.snr = (torch.ones(B, 1).to(xyz.device)) * self.test_SNR

        # xyz = xyz.permute(0, 2, 1)
        if self.normal_channel:
            feats = xyz.transpose(1, 2)
            l0_xyz = xyz[:, :, :3]
        else:
            feats = xyz.transpose(1, 2)
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.down1(l0_xyz, feats) #(32,512,3), (32,128,512)
        gt_downsample = l1_xyz
        _, g1_points = self.block0(l1_xyz, l1_points) #(32,512,3), (32,128,512)
        l2_xyz, l2_points = self.down2(l1_xyz, g1_points)
        _, g2_points = self.block1(l2_xyz, l2_points)

        # x = self.CAnet(g2_points)           # FCDR module
        x = g2_points
  
        x_send = F.adaptive_max_pool1d(x, 1).view(B, -1)
      
        ##################channel##########################
        
        received_points = self.channel(x_send, band)

        if band != self.bottleneck_size:
            mask_chanel = self.bottleneck_size -band
            zero_tensor = torch.zeros(received_points.shape[0], mask_chanel).to(received_points.device) 
            new_points = torch.cat([received_points, zero_tensor], dim=1)
        else:
            new_points = received_points

        
        # Initial Coordinate Estimation
        y = new_points.unsqueeze(1)                     #(1,128)
        decoder_local_feature = self.decompression(y)  #(B,128,128)
        new_xyz0 = self.coor_reconstruction_layer(decoder_local_feature.permute(0, 2, 1))  #(B,128,3)

        # CQDP and Offset-Based Coordinate Reconstruction
        coor_recon = self.net(new_xyz0.permute(0, 2, 1), decoder_local_feature, self.snr)   #coor_recon为三个tensor：原始点云下采样一次的坐标、重建的坐标、精细化后的坐标
        
        return coor_recon, gt_downsample


