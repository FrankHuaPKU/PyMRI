import numpy as np
import scipy.fft

from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import os

from .turbulence import ScalarField, VectorField, Turbulence

class CorrFunc(ScalarField):
    """关联函数类, 继承自ScalarField
    
    用于存储关联函数数据。实际上为一个实空间的标量场。
    
    对象: 标量场
    实例: 磁场关联函数, 速度关联函数
    
    属性:
        data (np.ndarray)    : 存储关联函数值(继承自ScalarField)
        Lx, Ly, Lz (float)   : box尺寸(继承自ScalarField)
        
    继承了ScalarField的所有运算
    """
    
    def __init__(self, data: np.ndarray, box: List[float]):
        """初始化关联函数场
        
        参数:
            data: 关联函数数据数组
            box: [Lx, Ly, Lz] 模拟box的三个方向尺寸
        """
        # 调用父类的初始化方法
        super().__init__(data, box)
        
    @property
    def normalize(self) -> 'CorrFunc':
        """将关联函数归一化(对0处关联函数值归一化)
        
        返回:
            归一化后的关联函数
        """
        return CorrFunc(self.data / self.data[self.Nx//2, self.Ny//2, self.Nz//2], self.box)


def get_corrfunc(field: VectorField) -> CorrFunc:
    """根据矢量场计算关联函数
    
    参数:
        field: VectorField类型的磁场或速度
        
    返回:
        CorrFunc: 计算得到的关联函数
    """
    # 计算三个分量的傅里叶变换
    fieldx_hat: np.ndarray = scipy.fft.fftn(field.x, workers=-1) / (field.Nx * field.Ny * field.Nz)
    fieldy_hat: np.ndarray = scipy.fft.fftn(field.y, workers=-1) / (field.Nx * field.Ny * field.Nz)
    fieldz_hat: np.ndarray = scipy.fft.fftn(field.z, workers=-1) / (field.Nx * field.Ny * field.Nz)
    
    # 计算总能谱
    spc: np.ndarray = np.abs(fieldx_hat)**2 + np.abs(fieldy_hat)**2 + np.abs(fieldz_hat)**2

    # 计算关联函数(能谱的逆傅里叶变换)
    corrfunc = scipy.fft.ifftn(spc, workers=-1)
    corrfunc = np.fft.fftshift(corrfunc)
    corrfunc = np.real(corrfunc)
    
    # 创建CorrFunc对象
    return CorrFunc(corrfunc, [field.Lx, field.Ly, field.Lz])


class Correlation:
    """关联函数类
    
    用于计算和存储MRI湍流场的磁场关联函数与速度关联函数。
    
    属性:
        turbulence (Turbulence)    : MRI湍流场对象
        normalized (bool)          : 是否需要归一化
        Bcorrfuncs (List[CorrFunc]): 磁场关联函数列表
        Vcorrfuncs (List[CorrFunc]): 速度关联函数列表
        times (List[float])        : 时间列表
        Bmean (CorrFunc)           : 磁场关联函数的平均值
        Bstd  (CorrFunc)           : 磁场关联函数的标准差
        Vmean (CorrFunc)           : 速度关联函数的平均值
        Vstd  (CorrFunc)           : 速度关联函数的标准差
    """
    
    def __init__(self, turbulence: Turbulence, normalized: bool = True):
        """初始化关联函数对象
        
        参数:
            turbulence: MRI湍流场对象
            normalized: 是否需要归一化
        """
        self.turbulence = turbulence
        self.normalized = normalized
        self.times = turbulence.times

        # 计算每个时刻的关联函数
        self.Bcorrfuncs = [get_corrfunc(B) for B in turbulence.Bs]
        self.Vcorrfuncs = [get_corrfunc(V) for V in turbulence.Vs]

        # 如果需要归一化，对每个关联函数进行归一化
        if normalized:
            self.Bcorrfuncs = [corrfunc.normalize for corrfunc in self.Bcorrfuncs]
            self.Vcorrfuncs = [corrfunc.normalize for corrfunc in self.Vcorrfuncs]

        # 计算平均值和标准差
        def get_mean_std(corrfuncs: List[CorrFunc]) -> Tuple[CorrFunc, CorrFunc]:
            """计算关联函数列表的平均值和标准差
            """
            # 提取box
            box = corrfuncs[0].box

            # 计算平均值
            stacked_data: np.ndarray = np.stack([corrfunc.data for corrfunc in corrfuncs], axis=0)
            mean_data   : np.ndarray = np.mean(stacked_data, axis=0)
            std_data    : np.ndarray = np.std( stacked_data, axis=0)

            # 创建CorrFunc对象
            mean: CorrFunc = CorrFunc(mean_data, box)
            std : CorrFunc = CorrFunc(std_data , box)

            return mean, std

        # 存储平均值和标准差
        self.Bmean, self.Bstd = get_mean_std(self.Bcorrfuncs)
        self.Vmean, self.Vstd = get_mean_std(self.Vcorrfuncs)

        # 用一个字典将var转换为title
        self.title_dict: Dict[str, str] = {
            'B': 'Magnetic Field Correlation Function',
            'V': 'Velocity Field Correlation Function'
        }

    def plot2d(self, var: str):
        """绘制2D关联函数切片图
        
        参数:
            var: 关联函数变量
                - 'B': 磁场关联函数
                - 'V': 速度关联函数
        """
        # 用一个字典将var转换为mean和std
        mean_dict = {'B': self.Bmean, 'V': self.Vmean}
        
        # 获取对应的平均关联函数
        mean: CorrFunc = mean_dict[var]
        
        # 创建输出目录
        output_dir = os.path.join('corrfunc', '2d')
        os.makedirs(output_dir, exist_ok=True)

        # 计算坐标
        x = np.linspace(-mean.Lx/2, mean.Lx/2, mean.Nx)
        y = np.linspace(-mean.Ly/2, mean.Ly/2, mean.Ny)
        z = np.linspace(-mean.Lz/2, mean.Lz/2, mean.Nz)

        # 提取三个平面的切片
        corrfunc_yx = mean.data[:, :, mean.Nz//2].T
        corrfunc_xz = mean.data[:, mean.Ny//2, :]
        corrfunc_yz = mean.data[mean.Nx//2, :, :]

        # 创建三个子图
        for plane, data, xlabel, ylabel in [
            ('yz', corrfunc_yz, 'y', 'z'),
            ('xz', corrfunc_xz, 'x', 'z'),
            ('yx', corrfunc_yx, 'y', 'x')
        ]:
            # 创建新图，不指定figsize，让实际比例由数据决定
            plt.figure()
            
            # 根据plane选择合适的坐标
            if plane == 'yz':
                X, Y = np.meshgrid(y, z, indexing='ij')
            elif plane == 'xz':
                X, Y = np.meshgrid(x, z, indexing='ij')
            else:  # plane == 'yx'
                X, Y = np.meshgrid(y, x, indexing='ij')

            # 绘制关联函数
            plt.pcolormesh(X, Y, data, cmap='jet', shading='auto')
            plt.colorbar(label='Correlation Function')

            # 设置标签和标题
            plt.xlabel(f'${xlabel}$', fontsize=12)
            plt.ylabel(f'${ylabel}$', fontsize=12)
            plt.title(f'{self.title_dict[var]}', fontsize=14)

            # 保持横纵轴等比例, 但图中横纵轴的范围无需相等
            plt.gca().set_aspect('equal', adjustable='box')
            

            # 调整布局以确保所有元素都能显示
            # plt.tight_layout()

            # 保存图像
            filename = f'{var}corrfunc({plane}).pdf'
            plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
            plt.close()

    def plot1d(self, var: str):
        """绘制1D关联函数切片图
        
        参数:
            var: 关联函数变量
                - 'B': 磁场关联函数
                - 'V': 速度关联函数
        """
        # 用一个字典将var转换为mean和std
        mean_dict = {'B': self.Bmean, 'V': self.Vmean}
        std_dict  = {'B': self.Bstd , 'V': self.Vstd}
        
        # 获取对应的平均关联函数和标准差
        mean = mean_dict[var]
        std  = std_dict[var]

        # 创建输出目录
        output_dir = os.path.join('corrfunc', '1d')
        os.makedirs(output_dir, exist_ok=True)

        # 计算坐标（只取正半轴部分）
        x = np.linspace(0, mean.Lx/2, mean.Nx//2)
        y = np.linspace(0, mean.Ly/2, mean.Ny//2)
        z = np.linspace(0, mean.Lz/2, mean.Nz//2)

        # 确定中心点索引
        center_x, center_y, center_z = mean.Nx//2, mean.Ny//2, mean.Nz//2

        # 提取三个方向的1D切片（只取正半轴部分）
        corrfunc_x = mean.data[center_x:, center_y , center_z ]
        corrfunc_y = mean.data[center_x , center_y:, center_z ]
        corrfunc_z = mean.data[center_x , center_y , center_z:]

        std_x = std.data[center_x:, center_y , center_z ]
        std_y = std.data[center_x , center_y:, center_z ]
        std_z = std.data[center_x , center_y , center_z:]

        # 为三个方向分别绘图
        for direction, r, corr, err in [
            ('x', x, corrfunc_x, std_x),
            ('y', y, corrfunc_y, std_y),
            ('z', z, corrfunc_z, std_z)
        ]:
            # 创建新图
            plt.figure(figsize=(7, 5))

            # 绘制关联函数和误差范围
            plt.plot(r, corr, 'k-', lw=2)
            plt.fill_between(r, corr - err, corr + err, color='gray', alpha=0.3)

            # 设置标签和标题
            plt.xlabel(f'${direction}$', fontsize=12)
            plt.ylabel('Correlation', fontsize=12)
            plt.title(f'{self.title_dict[var]}', fontsize=14)

            # 添加网格
            plt.grid(True, ls='--', lw=0.5)

            # 保存图像
            filename = f'{var}corrfunc({direction}).pdf'
            plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
            plt.close()

    def plot(self):
        """绘制所有关联函数图像"""
        # 绘制2D切片图
        self.plot2d(var='B')
        self.plot2d(var='V')

        # 绘制1D切片图
        self.plot1d(var='B')
        self.plot1d(var='V')
