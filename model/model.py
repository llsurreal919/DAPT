import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from paconv_util import Down, ReconstructionLayer, PointTransformerBlock
from pucrn_adaptive import CRNet


class Channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=8):
        """
        第一层全连接层神经元个数较少, 因此需要一个比例系数ratio进行缩放
        """
        super(Channel_attention, self).__init__()
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
        self.fc1   = nn.Conv1d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(in_channel // ratio, in_channel, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid() 
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class Model(nn.Module):
    def __init__(self, normal_channel=False, bottleneck_size=320, recon_points=2048, channel_name='AWGN',
                 SNR_MIN=0, SNR_MAX=10, SNR_val=5):
        super(Model, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.normal_channel = normal_channel
        self.recon_points = recon_points
    
        self.SNR_MAX = SNR_MAX
        self.SNR_MIN = SNR_MIN
        self.test_SNR = SNR_val

        self.channel_name = channel_name
        self.Downsample_PAconv1 = Down(in_channels=3, out_channels=128, stride=4, num_neighbors=16, scorenet_input_dim=7)
        self.Downsample_PAconv2 = Down(in_channels=128, out_channels=bottleneck_size, stride=4, num_neighbors=16, scorenet_input_dim=257)
        
        self.Point_Trans_Block1 = PointTransformerBlock(in_channels=128, num_neighbors=16)
        self.Point_Trans_Block2 = PointTransformerBlock(in_channels=bottleneck_size, num_neighbors=16)

        self.FCDR_net = Channel_attention(in_channel=bottleneck_size)
        
        self.decompression = ReconstructionLayer(recon_points // 16, bottleneck_size, 128)
        self.coor_reconstruction_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

 
        self.net = CRNet(up_ratio = 4)

       
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
            AWGN = (noise_std * torch.randn_like(data_received)).to(data_received.device)
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

    def forward(self, points, band, training = True, simu_name = 'SNR_adaptive_model'):
        B, N, _ = points.shape
        if training is True:
            if simu_name=='SNR_adaptive_model':
                self.snr =  (torch.rand(B, 1).to(points.device)) * (self.SNR_MAX-self.SNR_MIN) + self.SNR_MIN
            elif simu_name=='SNR_independent_model':
                self.snr= (torch.ones(B, 1).to(points.device)) * self.test_SNR
        else:
            self.snr = (torch.ones(B, 1).to(points.device)) * self.test_SNR

        if self.normal_channel:
            feats = points.transpose(1, 2)
            level_0_points = points[:, :, :3]
        else:
            feats = points.transpose(1, 2)
            level_0_points = points
        
        ## PFDA Module ##
        level_1_points, level_1_feats = self.Downsample_PAconv1(level_0_points, feats) # FPS + KNN + PAConv
        gt_downsample = level_1_points
        _, enhanced_level_1_feats = self.Point_Trans_Block1(level_1_points, level_1_feats) # point transformer
        level_2_points, level_2_feats = self.Downsample_PAconv2(level_1_points, enhanced_level_1_feats) # FPS + KNN + PAConv
        _, enchanced_level_2_feats = self.Point_Trans_Block2(level_2_points, level_2_feats) # point transformer

        ## FCDR module ##
        x = self.FCDR_net(enchanced_level_2_feats)          # x(batch, 320, 128),   320 is channel number, 128 is point number

        ## adopt maxpolling to further reduce the features size ##
        x_send = F.adaptive_max_pool1d(x, 1).view(B, -1)
      
        ## channel##
        received_points = self.channel(x_send, band)

        ## compensate before decode ##
        if band != self.bottleneck_size:
            mask_chanel = self.bottleneck_size - band
            zero_tensor = torch.zeros(received_points.shape[0], mask_chanel).to(received_points.device) 
            received_points = torch.cat([received_points, zero_tensor], dim=1)

        ## Initial Coordinate Estimation ##
        received_points = received_points.unsqueeze(1)  
        de_level_0_feature = self.decompression(received_points)  # T-Conv layer
        de_level_0_points = self.coor_reconstruction_layer(de_level_0_feature.permute(0, 2, 1))  # MLP layer

        # CQDP and OBCR Modules ##
        coor_recon = self.net(de_level_0_points.permute(0, 2, 1), de_level_0_feature, self.snr)   #coor_recon为三个tensor：原始点云下采样一次的坐标、重建的坐标、精细化后的坐标
        
        return coor_recon, gt_downsample


