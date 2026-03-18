import numpy as np
import torch
import os

def save_point_clouds_for_visualization(pc_original, pc_reconstructed, step, save_dir="/home/danny/WJJ/point_cloud_jscc/visualization"):
    """
    保存原始和重建点云为PLY文件，供专业软件查看
    """
    import open3d as o3d
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 转换Tensor到numpy
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
    
    # 4. 可选：添加颜色以便区分（专业软件中会显示）
    pcd_orig.paint_uniform_color([0, 0, 0])
    pcd_recon.paint_uniform_color([0, 0, 0])
    
    # 5. 保存为PLY文件（最通用的点云格式）
    orig_path = os.path.join(save_dir, f"step_{step:04d}_original.ply")
    recon_path = os.path.join(save_dir, f"step_{step:04d}_reconstructed.ply")
    
    o3d.io.write_point_cloud(orig_path, pcd_orig)
    o3d.io.write_point_cloud(recon_path, pcd_recon)
    
    print(f"✅ 点云已保存:")
    print(f"   原始点云: {orig_path}")
    print(f"   重建点云: {recon_path}")
    
    return orig_path, recon_path