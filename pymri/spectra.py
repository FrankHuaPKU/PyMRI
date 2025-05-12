import numpy as np
import scipy.fft

from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import os

from .turbulence import ScalarField, VectorField, Turbulence

class Spectrum(ScalarField):
    """能谱类, 继承自ScalarField
    
    用于存储能谱数据。实际上为一个谱空间的标量场。
    
    对象: 标量场
    实例: 磁场能谱, 动能能谱, 磁耗散谱, 黏性耗散谱
    
    属性:
        data (np.ndarray)    : 存储能谱值(继承自ScalarField)
        Lx, Ly, Lz (float)   : box尺寸(继承自ScalarField)
        dkx, dky, dkz (float): 谱空间网格间隔, 等于模拟box的x,y,z方向尺寸的倒数 1/Lx, 1/Ly, 1/Lz
        dVk (float)          : 谱空间体积元, 等于 dkx * dky * dkz
        
    继承了ScalarField的所有运算
    """
    
    def __init__(self, spc: np.ndarray, box: List[float]):
        """初始化能谱场
        
        参数:
            spc: 能谱数据数组
            box: [Lx, Ly, Lz] 模拟box的三个方向尺寸
        """
        # 调用父类的初始化方法
        super().__init__(spc, box)
        
        # 计算谱空间特有的属性
        self.dkx = 1.0 / self.Lx
        self.dky = 1.0 / self.Ly
        self.dkz = 1.0 / self.Lz
        self.dVk = self.dkx * self.dky * self.dkz
        
        # 生成三个方向的波数
        self.kx = np.fft.fftfreq(self.Nx, d=self.dx)
        self.ky = np.fft.fftfreq(self.Ny, d=self.dy)
        self.kz = np.fft.fftfreq(self.Nz, d=self.dz)


class MagneticSpectra:
    """磁场能谱类
    
    用于计算和存储MRI湍流场的磁场能谱。
    
    属性:
        turbulence (Turbulence) : MRI湍流场对象
        normalized (bool)       : 是否需要归一化, 默认为False
        Bs (List[VectorField])  : 磁场列表, 从turbulence中提取
        times (List[float])     : 时间列表, 从turbulence中提取
        spectra (List[Spectrum]): 磁场能谱列表
    """
    
    def __init__(self, turbulence: Turbulence, normalized: bool = False):
        """初始化磁场能谱对象
        
        参数:
            turbulence: MRI湍流场对象
            normalized: 是否需要归一化, 默认为False
        """
        self.turbulence = turbulence
        self.normalized = normalized
        
        # 从turbulence中提取数据
        self.Bs: List[VectorField] = turbulence.Bs
        self.times: List[float]    = turbulence.times
        
    @property
    def spectra(self) -> List[Spectrum]:
        """计算磁场能谱列表, 并存储为类属性
        
        返回:
            List[Spectrum]: 磁场能谱列表
        """
        def spectrum(B: VectorField) -> Spectrum:
            """根据磁场计算能谱
            
            参数:
                B: VectorField类型的磁场
                
            返回:
                Spectrum: 计算得到的能谱
            """
            # 计算三个分量的傅里叶变换
            Bx_hat: np.ndarray = scipy.fft.fftn(B.x, workers=-1) / (B.Nx * B.Ny * B.Nz)
            By_hat: np.ndarray = scipy.fft.fftn(B.y, workers=-1) / (B.Nx * B.Ny * B.Nz)
            Bz_hat: np.ndarray = scipy.fft.fftn(B.z, workers=-1) / (B.Nx * B.Ny * B.Nz)
            
            # 计算总能谱
            spc: np.ndarray = 0.5 * (np.abs(Bx_hat)**2 + np.abs(By_hat)**2 + np.abs(Bz_hat)**2)
            
            # 创建Spectrum对象
            return Spectrum(spc, [B.Lx, B.Ly, B.Lz])
        
        # 计算所有时刻的能谱
        spectra: List[Spectrum] = [spectrum(B) for B in self.Bs]
        
        # 如果需要归一化，则除以对应时刻的总磁能
        if self.normalized:
            MEs: List[float] = self.turbulence.MEs
            spectra = [spc * (1.0 / ME) for spc, ME in zip(spectra, MEs)]
            
        return spectra

    @property
    def avg(self) -> Spectrum:
        """计算平均能谱
        
        返回:
            Spectrum: 平均能谱
        """
        # 获取所有能谱
        spectra = self.spectra
        
        if not spectra:
            raise ValueError("没有可用的能谱数据")
            
        # 获取第一个能谱的box信息
        box = [spectra[0].Lx, spectra[0].Ly, spectra[0].Lz]
        
        # 将所有能谱数据堆叠成三维数组，然后沿第一维取平均
        data_stack: np.ndarray = np.stack([spc.data for spc in spectra])
        avg_data: np.ndarray   = np.mean(data_stack, axis=0)
        
        return Spectrum(avg_data, box)

    def plot2d(self, output_dir: str = 'spectra', filename: str = 'avgMEspectra2D.pdf'):
        """绘制2D能谱
        
        参数:
            output_dir: 输出目录，默认为'spectra'
            filename: 输出文件名，默认为'avgMEspectra2D.pdf'
        """
        # 获取平均能谱
        spectrum: Spectrum = self.avg
        
        # 使用fftshift将频率零点移到中心
        kx: np.ndarray = np.fft.fftshift(spectrum.kx)
        ky: np.ndarray = np.fft.fftshift(spectrum.ky)
        kz: np.ndarray = np.fft.fftshift(spectrum.kz)
        Ek: np.ndarray = np.fft.fftshift(spectrum.data)
        
        # 提取三种能谱切片
        Ek_xy: np.ndarray = Ek[:, :, spectrum.Nz // 2]    # E(kx, ky, 0)
        Ek_xz: np.ndarray = Ek[:, spectrum.Ny // 2, :]    # E(kx, 0, kz)
        Ek_zy: np.ndarray = Ek[spectrum.Nx // 2, :, :].T  # E(0, ky, kz)
        
        # 计算全局vmin和vmax
        all_Ek = np.concatenate([Ek_xy.flatten(), Ek_xz.flatten(), Ek_zy.flatten()])
        vmin: float = all_Ek.min()
        vmax: float = all_Ek.max()
        
        # 创建统一的LogNorm
        norm: LogNorm = LogNorm(vmin=vmin, vmax=vmax)
        
        # 创建绘图，定义子图位置
        fig = plt.figure(figsize=(11, 5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.07], height_ratios=[1, 1], wspace=0.1, hspace=0.)
        
        # 左上子图：kz=0, kx vs ky
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        ax1 = fig.add_subplot(gs[0, 0])
        pcm1 = ax1.pcolormesh(KX, KY, Ek_xy, norm=norm, cmap='tab20c', shading='auto')
        ax1.set_xlabel(r'$k_x$', fontsize=12, labelpad=5)
        ax1.set_ylabel(r'$k_y$', fontsize=12, labelpad=0)
        ax1.xaxis.set_label_position('top')
        ax1.xaxis.tick_top()
        ax1.yaxis.set_label_position('left')
        ax1.yaxis.tick_left()
        ax1.set_aspect('equal')
        
        # 左下子图：kx=0, kz vs ky
        KZ, KY = np.meshgrid(kz, ky, indexing='ij')
        ax2 = fig.add_subplot(gs[1, 0])
        pcm2 = ax2.pcolormesh(KZ, KY, Ek_zy, norm=norm, cmap='tab20c', shading='auto')
        ax2.set_xlabel(r'$k_z$', fontsize=12, labelpad=5)
        ax2.set_ylabel(r'$k_y$', fontsize=12, labelpad=0)
        ax2.xaxis.set_label_position('bottom')
        ax2.xaxis.tick_bottom()
        ax2.yaxis.set_label_position('left')
        ax2.yaxis.tick_left()
        ax2.set_aspect('equal')
        
        # 右侧子图：ky=0, kx vs kz
        KX, KZ = np.meshgrid(kx, kz, indexing='ij')
        ax3 = fig.add_subplot(gs[:, 1])
        pcm3 = ax3.pcolormesh(KX, KZ, Ek_xz, norm=norm, cmap='tab20c', shading='auto')
        ax3.set_xlabel(r'$k_x$', fontsize=12, labelpad=5)
        ax3.set_ylabel(r'$k_z$', fontsize=12, labelpad=0)
        ax3.xaxis.set_label_position('bottom')
        ax3.xaxis.tick_bottom()
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.tick_right()
        ax3.set_aspect('equal')
        
        # 添加colorbar
        cbar = fig.colorbar(pcm3, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.0215, pad=0.1)
        cbar.set_label('$E(k)$', fontsize=12)
        
        # 设置标题
        plt.title('Magnetic Energy Spectrum', fontsize=14)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()


