import collections
import random
import open3d
import os
import numpy as np
import torch
from utils.dataset import Dataset
from torch.utils.data import DataLoader
from utils.pc_error_wrapper import pc_error
import time
import importlib
import sys
import argparse
import logging
from logging import handlers
from thop import profile
import open3d as o3d
from utils.save_pointcloud import save_point_clouds_for_visualization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def visualize_tensors(pc_original, pc_reconstructed, step=None):
    """
    将两个点云Tensor在同一位置可视化对比
    原始点云和重建点云在同一坐标系中，用不同颜色区分
    点云显示为小黑圆点
    """
    # 1. 确保Tensor在CPU上并转为numpy
    if torch.is_tensor(pc_original):
        pc_original = pc_original.cpu().numpy()
    if torch.is_tensor(pc_reconstructed):
        pc_reconstructed = pc_reconstructed.cpu().numpy()
    
    # 2. 处理批量维度（取第一个样本）
    if pc_original.ndim == 3:
        pc_original = pc_original[0]
    if pc_reconstructed.ndim == 3:
        pc_reconstructed = pc_reconstructed[0]
    
    # 3. 创建Open3D点云对象
    pcd_orig = o3d.geometry.PointCloud()
    pcd_recon = o3d.geometry.PointCloud()
    
    pcd_orig.points = o3d.utility.Vector3dVector(pc_original)
    pcd_recon.points = o3d.utility.Vector3dVector(pc_reconstructed)
    
    # 4. 设置不同颜色以便区分
    # 原始点云：深灰色（接近黑）的小圆点
    pcd_orig.paint_uniform_color([0,0,0])  # 深灰色
    # 重建点云：红色小圆点（便于对比）
    pcd_recon.paint_uniform_color([0,0,0])  # 红色

    pcd_recon.translate([3.0, 0, 0])
    
    # 5. 创建可视化窗口并添加几何体
    vis = o3d.visualization.Visualizer()
    window_title = f"Step {step}" if step is not None else "Point Cloud Comparison"
    
    try:
        vis.create_window(window_name=window_title, width=800, height=600)
        vis.add_geometry(pcd_orig)
        vis.add_geometry(pcd_recon)  # 不平移，直接在同一位置添加
        
        # 6. 设置渲染选项使点云显示为小黑圆点
        # 注意：get_render_option()需要在添加几何体后调用
        render_opt = vis.get_render_option()
        if render_opt is not None:
            render_opt.point_size = 3.0  # 设置点的大小，可以根据需要调整
            render_opt.background_color = np.asarray([1.0, 1.0, 1.0])  # 白色背景
            render_opt.light_on = True  # 开启光照，使点更圆润
        else:
            print("警告: 无法获取渲染选项，点的大小和背景颜色可能使用默认设置")
        
        # 7. 设置视图控制，使两个点云在同一视角下
        view_ctl = vis.get_view_control()
        # 可以设置一些默认视角参数（可选）
        
        # 8. 运行可视化
        print(f"可视化窗口已打开... (原始: 深灰色点, 重建: 红色点)")
        print("提示: 关闭窗口继续运行")
        vis.run()
    except Exception as e:
        print(f"可视化失败: {e}")
        # 备选方案：使用matplotlib
        print("尝试使用matplotlib方案...")
        # visualize_with_matplotlib(pc_original, pc_reconstructed, step=step)
    finally:
        vis.destroy_window()

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args(band):
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('point_based_training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--dataset_path', type=str, default='/home/danny/WJJ/paconv_jscc/dataset')
    parser.add_argument('--model', default='model', help='model name [default: ]')
    parser.add_argument('--SNR_MIN', type=int, default=0, help='Signal to Noise Ratio')
    parser.add_argument('--SNR_MAX', type=int, default=10, help='Signal to Noise Ratio')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--channel_name', default='AWGN', help='channel name [default: AWGN] or Rayleigh')
    parser.add_argument('--seed', default=1, type=int, help='set seed')
    parser.add_argument('--band', default=band, type=int)
    parser.add_argument('--bottleneck_size', default=320, type=int)
    parser.add_argument('--recon_points', default=2048, type=int)
    
    return parser.parse_args()

def cal_d1(pc_gt, decoder_output, step, checkpoint_path):
    # 原始点云写入ply文件
    ori_pcd = open3d.geometry.PointCloud()  # 定义点云 2048个点
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(pc_gt))  # 定义点云坐标位置[N,3]
    orifile = checkpoint_path + '/temp_pc_file/' + 'd1_ori_' + str(step) + '.ply'  # 保存路径
    open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)
    # 重建点云写入ply文件
    rec_pcd = open3d.geometry.PointCloud()
    rec_pcd.points = open3d.utility.Vector3dVector(np.squeeze(decoder_output))
    recfile = checkpoint_path + '/temp_pc_file/' + 'd1_rec_' + str(step) + '.ply'
    open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

    pc_error_metrics = pc_error(infile1=orifile, infile2=recfile, res=2)  # res为数据峰谷差值
    pc_errors = [pc_error_metrics["mse1,PSNR (p2point)"][0],
                 pc_error_metrics["mse2,PSNR (p2point)"][0],
                 pc_error_metrics["mseF,PSNR (p2point)"][0],
                 pc_error_metrics["mse1      (p2point)"][0],
                 pc_error_metrics["mse2      (p2point)"][0],
                 pc_error_metrics["mseF      (p2point)"][0]]
    try:
        os.remove(orifile)
        os.remove(recfile)
    except OSError as e:
        print(f"删除临时文件失败: {e.strerror}")

    return pc_errors

def cal_d2(pc_gt, decoder_output, step, checkpoint_path):
    # 原始点云写入ply文件
    ori_pcd = open3d.geometry.PointCloud()  # 定义点云
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(pc_gt))  # 定义点云坐标位置[N,3]
    ori_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))  # 计算normal
    orifile = checkpoint_path + '/temp_pc_file/' + 'd2_ori_' + str(step) + '.ply'  # 保存路径
    print(orifile)
    open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)
    # 将ply文件中normal类型double转为float32
    lines = open(orifile).readlines()
    to_be_modified = [7, 8, 9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double', 'float32')
    file = open(orifile, 'w')
    for line in lines:
        file.write(line)
    file.close()
    # 可视化点云,only xyz
    # open3d.visualization.draw_geometries([ori_pcd])

    # 重建点云写入ply文件
    rec_pcd = open3d.geometry.PointCloud()
    rec_pcd.points = open3d.utility.Vector3dVector(np.squeeze(decoder_output))
    # rec_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30)) # 计算normal
    recfile = checkpoint_path + '/temp_pc_file/' + 'd2_rec_' + str(step) + '.ply'
    open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

    pc_error_metrics = pc_error(infile1=orifile, infile2=recfile, normal=True, res=2)  # res为数据峰谷差值,normal=True为d2
    pc_errors = [pc_error_metrics["mse1,PSNR (p2plane)"][0],
                 pc_error_metrics["mse2,PSNR (p2plane)"][0],
                 pc_error_metrics["mseF,PSNR (p2plane)"][0],
                 pc_error_metrics["mse1      (p2plane)"][0],
                 pc_error_metrics["mse1      (p2plane)"][0],
                 pc_error_metrics["mse1      (p2plane)"][0],
                 pc_error_metrics["mse2      (p2plane)"][0],
                 pc_error_metrics["mseF      (p2plane)"][0]]
    try:
        os.remove(orifile)
        os.remove(recfile)
    except OSError as e:
        print(f"删除临时文件失败: {e.strerror}")

    return pc_errors

def main(args, checkpoint, test_snr):
    model_path = './test_results/band_and_SNR_adaptive/band=' + str(args.band)

    checkpoint_path = model_path

    test_data = Dataset(root=args.dataset_path, dataset_name='shapenetcorev2', num_points=2048, split='val')
    test_loader = DataLoader(test_data, num_workers=2, batch_size=1, shuffle=False)
    
    avg_d1_psnr = np.array([0.0 for i in range(55)])
    avg_d1_mse = np.array([0.0 for i in range(55)])
    avg_d2_psnr = np.array([0.0 for i in range(55)])
    avg_d2_mse = np.array([0.0 for i in range(55)])
    counter = np.array([0.0 for i in range(55)])
    
    total_d1_psnr = 0.0
    total_d1_mse = 0.0
    total_d2_psnr = 0.0
    total_d2_mse = 0.0

    num_samples = 0

    if not os.path.exists(checkpoint_path + '/temp_pc_file'):
        os.makedirs(checkpoint_path + '/temp_pc_file')
    log = Logger(model_path + '/result.txt', level='debug')

    model = importlib.import_module(args.model).Model
    model = model(normal_channel=args.use_normals, bottleneck_size=args.bottleneck_size,
                    recon_points=args.recon_points, channel_name=args.channel_name,
                      SNR_MIN=args.SNR_MIN, SNR_MAX=args.SNR_MAX, SNR_val=test_snr)
    if not args.use_cpu:
        model = model.cuda()
    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])


    for step, data in enumerate(test_loader):      
        with torch.no_grad():
            pc_data = data[0]     #[1,2048,3]
            label = data[1]         #[1,1]

            if torch.cuda.is_available():
                pc_gt = pc_data.cuda()
                pc_data = pc_data.cuda()

            decoder_output, _= model(pc_data, args.band, training = False)  # coor_recon
            
            # if step >= 10:       # this pointcloud is a plan
            #     # visualize_tensors(pc_gt, decoder_output, step=step)
            #     save_point_clouds_for_visualization(pc_gt, decoder_output, step)
            #     xx = 1

            # 转换成numpy
            pc_gt = pc_gt.cpu().detach().numpy()
            decoder_output = decoder_output.cpu().detach().numpy()

            d1_results = cal_d1(pc_gt, decoder_output, step, checkpoint_path)
            d1_psnr = d1_results[2].item()
            d1_mse = d1_results[5].item()
            avg_d1_mse[label] += d1_mse
            total_d1_mse += d1_mse
            avg_d1_psnr[label] += d1_psnr
            total_d1_psnr += d1_psnr
            
            
            # D2 psnr & D2 mse
            d2_results = cal_d2(pc_gt, decoder_output, step, checkpoint_path)
            d2_psnr = d2_results[2].item()
            d2_mse = d2_results[7].item()
            avg_d2_mse[label] += d2_mse
            total_d2_mse += d2_mse
            avg_d2_psnr[label] += d2_psnr
            total_d2_psnr += d2_psnr

            log.logger.info(f"step: {step}")  
            # log.logger.info(f"chamfer_distance: {cd}")         
            log.logger.info(f"d1_psnr: {d1_psnr}")
            log.logger.info(f"d2_psnr: {d2_psnr}")
            log.logger.info("-----------------------------------------------------")

        counter[label] += 1
        num_samples += 1

    for i in range(55):
        avg_d1_psnr[i] /= counter[i]
        avg_d1_mse[i] /= counter[i]
        avg_d2_psnr[i] /= counter[i]
        avg_d2_mse[i] /= counter[i]
    
    total_d1_psnr /= num_samples
    total_d1_mse /= num_samples
    total_d2_psnr /= num_samples
    total_d2_mse /= num_samples

    for i in range(55):
        outstr = str(i) + "Average_D1_PSNR: %.6f, Average_D1_mse: %.6f, Average_D2_PSNR: %.6f, Average_D2_mse: %.6f\n" % (        
            avg_d1_psnr[i], avg_d1_mse[i], avg_d2_psnr[i], avg_d2_mse[i])
        log.logger.info(f"{outstr}")

    outstr = "Total_D1_PSNR: %.6f, Total_D1_mse: %.6f, Total_D2_PSNR: %.6f, Total_D2_mse: %.6f\n" % (
        total_d1_psnr, total_d1_mse, total_d2_psnr, total_d2_mse)
    log.logger.info(f"{outstr}")

    return total_d1_psnr


if __name__ == '__main__':

    PSNR = np.zeros([4, 11])
    bands = [64, 128, 200, 256]
    for idx, band in enumerate(bands):
        args = parse_args(band)

        seed_everything(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        checkpoint_path ='Checkpoints/(Rayleigh)Band_and_SNR_adaptive/180.pth'
        checkpoint = torch.load(checkpoint_path)
        psnr_list = []
        
        for test_snr in range(0, 11):
            psnr = main(args, checkpoint, test_snr)
            psnr_list.append(psnr)
        PSNR[idx, :] = psnr_list
    print(PSNR)



   

    
    
