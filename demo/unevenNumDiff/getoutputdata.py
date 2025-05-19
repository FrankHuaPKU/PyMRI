#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 提取用于估计数值耗散的数据
# 连续输出一系列时刻, 例如33个时刻, 以某一周期性时刻(记作t0)为中心, 用于测试 2, 4, 8, 16, 32 阶差分

# 设模拟时间步长大致为 dt(一般取上限), 则需要提取 t0 - 16 * dt, t0 + 16 * dt 范围内数据
# 按照该原则指定输出数据(id设置为ndiss, 代表数值耗散)

# 本脚本需要读取 case 目录中的所有 outputs/HGB.ndiss.*.athdf 文件
# 读取这些文件中的数据, 利用 preprocess.py 提取物理场并随机选择 N 个空间点
# 绘制这 N 点处的各个物理量(包括 Vx, Vy, Vz, Bx, By, Bz)随时间变化的图像, 每一张图对应一个物理量

# 在一个case目录中运行本脚本, 按照相对路径: python ../../../PyMRI/demo/unevenNumDiff/getoutputdata.py
# 输出结果为包括一系列.pdf图片文件: Vx.pdf, Vy.pdf, Vz.pdf, Bx.pdf, By.pdf, Bz.pdf
# 输出文件存储于 case 目录中的 test_ndiss 子目录中


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# 添加PyMRI库的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
ATHENAUI_ROOT = os.path.abspath(os.path.join(script_dir, '../../..'))
sys.path.insert(0, ATHENAUI_ROOT)

# 导入preprocess模块
from src.post.preprocess import output2turbulence

def select_random_points(shape: Tuple[int, int, int], N: int = 10) -> List[Tuple[int, int, int]]:
    """随机选择N个空间点
    
    参数:
        shape: 数据场的形状 (nx, ny, nz)
        N: 要选择的点数
        
    返回:
        List[Tuple[int, int, int]]: N个随机点的坐标
    """
    nx, ny, nz = shape
    points = []
    for _ in range(N):
        x = np.random.randint(0, nx)
        y = np.random.randint(0, ny)
        z = np.random.randint(0, nz)
        points.append((x, y, z))
    return points

def extract_point_data(turbulence, points: List[Tuple[int, int, int]]) -> Dict:
    """提取选定点的时间序列数据
    
    参数:
        turbulence: Turbulence对象
        points: 选定的空间点列表
        
    返回:
        Dict: 包含各个物理量在选定点的时间序列数据
    """
    times = turbulence.times
    data = {
        'Vx': [], 'Vy': [], 'Vz': [],
        'Bx': [], 'By': [], 'Bz': [],
        'times': times
    }
    
    for t in range(len(times)):
        V = turbulence.Vs[t]
        B = turbulence.Bs[t]
        
        # 提取每个点的数据
        vx_points = [V.x[p[0], p[1], p[2]] for p in points]
        vy_points = [V.y[p[0], p[1], p[2]] for p in points]
        vz_points = [V.z[p[0], p[1], p[2]] for p in points]
        
        bx_points = [B.x[p[0], p[1], p[2]] for p in points]
        by_points = [B.y[p[0], p[1], p[2]] for p in points]
        bz_points = [B.z[p[0], p[1], p[2]] for p in points]
        
        data['Vx'].append(vx_points)
        data['Vy'].append(vy_points)
        data['Vz'].append(vz_points)
        data['Bx'].append(bx_points)
        data['By'].append(by_points)
        data['Bz'].append(bz_points)
    
    return data

def plot_time_series(data: Dict, output_dir: str):
    """绘制时间序列图
    
    参数:
        data: 包含时间序列数据的字典
        output_dir: 输出目录
    """
    times = data['times']
    fields = ['Vx', 'Vy', 'Vz', 'Bx', 'By', 'Bz']
    
    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    for field in fields:
        plt.figure(figsize=(12, 8))
        field_data = np.array(data[field])  # shape: (nt, npoints)
        
        # 绘制每个点的时间序列，全部使用黑色加粗线条
        for i in range(field_data.shape[1]):
            plt.plot(times, field_data[:, i], 'k-', linewidth=2)
        
        plt.xlabel('Time', labelpad=10)
        plt.ylabel(field, labelpad=10)
        plt.title(f'Time Evolution of {field}', pad=15)
        plt.grid(True)
        
        # 保存图像
        output_file = os.path.join(output_dir, f'{field}.pdf')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f'已保存图像: {output_file}')

def main():
    # 创建输出目录
    output_dir = 'test_ndiss'
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取turbulence数据
    # 注意：这里的时间范围需要根据实际情况调整
    turbulence = output2turbulence('ndiss', 0, None)
    if turbulence is None:
        print('错误: 无法提取数据')
        return
    
    # 获取数据场形状并选择随机点
    shape = turbulence.Vs[0].x.shape
    points = select_random_points(shape, N=10)
    
    # 提取选定点的数据
    data = extract_point_data(turbulence, points)
    
    # 绘制并保存图像
    plot_time_series(data, output_dir)
    
    print('\n数据处理完成!')

if __name__ == '__main__':
    main()