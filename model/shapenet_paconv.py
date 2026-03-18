import torch.nn as nn
import torch.nn.functional as F


import torch
from pytorch3d.loss import chamfer_distance
from paconv_util import Embedding, ReconstructionLayer, PointTransformerBlock
from shapenet_utils import TransitionDown
from pucrn_adaptive import CRNet



class get_model(nn.Module):
    def __init__(self, normal_channel=False, bottleneck_size=200, recon_points=2048, SNR_MIN=0, SNR_MAX=10, SNR=5):
        super(get_model, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.normal_channel = normal_channel
        self.recon_points = recon_points
  
        
        
        self.SNR_MAX = SNR_MAX
        self.SNR_MIN = SNR_MIN
        self.test_SNR = SNR
       
       
        # self.down2 = Down(in_channels=128, out_channels=bottleneck_size, stride=4, num_neighbors=16, scorenet_input_dim=257)
        self.embedding = Embedding(in_channels=64, out_channels=64, scorenet_input_dim=6)
        self.down1 = TransitionDown(64, 128, stride=4, num_neighbors=16)
        self.down2 = TransitionDown(128, bottleneck_size, stride=4, num_neighbors=16)
        
        self.block0 = PointTransformerBlock(in_channels=128, num_neighbors=16)
        self.block1 = PointTransformerBlock(in_channels=bottleneck_size, num_neighbors=16)
        
        
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
    
    def channel(self, data, get_snr):
        with torch.no_grad():
            noise_std = 10 ** (-get_snr * 1.0 / 20)
            AWGN = (noise_std * torch.randn_like(data)).to(data.device)
        data_r = data + AWGN
        return data_r
        
    

    def forward(self, xyz, isTrain):
        # xyz = xyz.permute(0, 2, 1)
        B, N, _ = xyz.shape
        if self.normal_channel:
           
            l0_xyz = xyz[:, :, :3]
        else:
            l0_xyz = xyz
        
        l0_feats = self.embedding(l0_xyz)  #(32,64,2048)
        
        l1_xyz, l1_points = self.down1(l0_xyz, l0_feats) #(32,512,3), (32,128,512)
        gt_downsample = l1_xyz
        _, g1_points = self.block0(l1_xyz, l1_points) #(32,512,3), (32,128,512)
        l2_xyz, l2_points = self.down2(l1_xyz, g1_points)
        _, g2_points = self.block1(l2_xyz, l2_points)
  
        x = F.adaptive_max_pool1d(g2_points, 1).view(B, -1)
        
      
        ##########channel
        
        z_out = self.power_norm(x)
        # z_out_power = torch.mean(z_out**2)    power = 1
        if isTrain is True:
            self.snr =  (torch.rand(B, 1).to(z_out.device)) * (self.SNR_MAX-self.SNR_MIN) + self.SNR_MIN
        else:
            self.snr = (torch.ones(B, 1).to(z_out.device)) * self.test_SNR
        new_points = self.channel(z_out, self.snr)
        
        
        y = new_points.unsqueeze(1)
        decoder_local_feature = self.decompression(y)  #(B,C256,128N)
        new_xyz0 = self.coor_reconstruction_layer(decoder_local_feature.permute(0, 2, 1))  #(B,N,3)
        coor_recon = self.net(new_xyz0.permute(0, 2, 1), decoder_local_feature, self.snr)
        
     
        
        return coor_recon, gt_downsample
       


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, coor_recon, pc_gd, gt_downsample=None, training=False):
        
        if (training is True):
            P1, P2, P3  = coor_recon
            cd_1 = chamfer_distance(P1, gt_downsample)[0]
            cd_2 = chamfer_distance(P2, pc_gd)[0]
            cd_3 = chamfer_distance(P3, pc_gd)[0]
            loss = cd_1 + cd_2 + cd_3 
        else:
            loss = chamfer_distance(coor_recon, pc_gd)[0]
            
        return loss

