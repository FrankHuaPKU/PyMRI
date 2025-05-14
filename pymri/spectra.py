import numpy as np
import scipy.fft

from scipy.stats import gmean

from typing import Tuple, List, Optional, Union, Dict

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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


class EnergySpectra:
    """能谱类
    
    用于计算和存储MRI湍流场的动能谱与磁能谱。
    
    属性:
        turbulence (Turbulence)   : MRI湍流场对象
        Bs (List[VectorField])    : 磁场列表, 从turbulence中提取
        wVs (List[VectorField])   : 加权速度列表, 从turbulence中提取
        times (List[float])       : 时间列表, 从turbulence中提取
        MEspectra (List[Spectrum]): 磁场能谱列表
        KEspectra (List[Spectrum]): 动能能谱列表
        MEgmean (Spectrum)        : 磁场能谱的几何平均值
        MEgstd  (Spectrum)        : 磁场能谱的几何标准差
        KEgmean (Spectrum)        : 动能能谱的几何平均值
        KEgstd  (Spectrum)        : 动能能谱的几何标准差
    """
    
    def __init__(self, turbulence: Turbulence):
        """初始化能谱对象
        
        参数:
            turbulence: MRI湍流场对象
        """
        self.turbulence = turbulence
        
        # 从turbulence中提取数据
        self.Bs : List[VectorField] = turbulence.Bs
        self.wVs: List[VectorField] = turbulence.wVs
        self.times: List[float]     = turbulence.times


        def get_spectrum(field: VectorField) -> Spectrum:
            """根据磁场计算能谱, 用于创建属性spectra
            
            参数:
                field: VectorField类型的磁场或加权速度
                
            返回:
                Spectrum: 计算得到的能谱
            """
            # 计算三个分量的傅里叶变换
            fieldx_hat: np.ndarray = scipy.fft.fftn(field.x, workers=-1) / (field.Nx * field.Ny * field.Nz)
            fieldy_hat: np.ndarray = scipy.fft.fftn(field.y, workers=-1) / (field.Nx * field.Ny * field.Nz)
            fieldz_hat: np.ndarray = scipy.fft.fftn(field.z, workers=-1) / (field.Nx * field.Ny * field.Nz)
            
            # 计算总能谱
            spc: np.ndarray = 0.5 * (np.abs(fieldx_hat)**2 + np.abs(fieldy_hat)**2 + np.abs(fieldz_hat)**2)
            
            # 创建Spectrum对象
            return Spectrum(spc, [field.Lx, field.Ly, field.Lz])

        self.MEspectra: List[Spectrum] = [get_spectrum(B)  for B  in self.Bs]
        self.KEspectra: List[Spectrum] = [get_spectrum(wV) for wV in self.wVs]


        def get_gmean_gstd(spectra: List[Spectrum]) -> Tuple[Spectrum, Spectrum]:
            """计算几何平均值和几何标准差
            
            参数:
                spectra: 能谱列表
                
            """
            # 提取box
            box: List[float] = [spectra[0].Lx, spectra[0].Ly, spectra[0].Lz]

            # 计算几何平均值, 即对数空间中的算数平均值
            stacked_data: np.ndarray = np.stack([spectrum.data for spectrum in spectra], axis=0)
            gmean_data  : np.ndarray = np.exp(np.mean(np.log(stacked_data), axis=0))

            # 计算几何标准差, 即对数空间中的标准差
            stacked_data: np.ndarray = np.stack([spectrum.data for spectrum in spectra], axis=0)
            gstd_data   : np.ndarray = np.exp(np.std(np.log(stacked_data), axis=0))

            # 创建Spectrum对象
            gmean: Spectrum = Spectrum(gmean_data, box)
            gstd : Spectrum = Spectrum(gstd_data , box)

            return gmean, gstd

        # 存储磁场能谱的几何平均值与几何标准差
        self.MEgmean, self.MEgstd = get_gmean_gstd(self.MEspectra)
        self.KEgmean, self.KEgstd = get_gmean_gstd(self.KEspectra)


        # 用一个字典将var转换为title
        self.title_dict: Dict[str, str] = {
            'ME': 'Magnetic Energy Spectrum',
            'KE': 'Kinetic Energy Spectrum'
        }

    def plot2d(self, var: str):
        """绘制2D能谱
        
        参数:
            var: 能谱变量
                - 'ME': 磁场能谱
                - 'KE': 动能能谱
        """

        # 用一个字典将var转换为spc
        gmean_dict: Dict[str, Spectrum] = {
            'ME': self.MEgmean,
            'KE': self.KEgmean
        }

        # 获取对应的平均能谱
        spc: Spectrum = gmean_dict[var]

        # 使用fftshift将频率零点移到中心
        kx: np.ndarray = np.fft.fftshift(spc.kx)
        ky: np.ndarray = np.fft.fftshift(spc.ky)
        kz: np.ndarray = np.fft.fftshift(spc.kz)
        Ek: np.ndarray = np.fft.fftshift(spc.data)
        
        # 提取三方向能谱切片
        # 提取平面附近几个（例如5个, 相当于覆盖0-2/H的波数空间）2D切片数据, 并取平均
        Ek_xy: np.ndarray = np.mean(Ek[:, :, spc.Nz // 2 - 2: spc.Nz // 2 + 3], axis=2)
        Ek_xz: np.ndarray = np.mean(Ek[:, spc.Ny // 2 - 2: spc.Ny // 2 + 3, :], axis=1)
        Ek_zy: np.ndarray = np.mean(Ek[spc.Nx // 2 - 2: spc.Nx // 2 + 3, :, :], axis=0).T
        
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
        plt.title(self.title_dict[var], fontsize=14)
        
        # 创建输出目录
        output_dir: str = 'spectra'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像
        save_path = os.path.join(output_dir, f'avg{var}spectra2D.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def plot1d(self, var: str, mode: str) -> None:
        """绘制一维能谱切片

        参数:
            var (str): 能谱变量
                - 'ME': 磁场能谱
                - 'KE': 动能能谱
            mode (str): 绘图模式
                - 'slice': 绘制三个方向的切片能谱，带标准差范围
                - 'times': 绘制z方向能谱随时间的演化

        输出文件:
            - mode == 'slice': {output_dir}/avg{var}spectraKx.pdf等三个文件
            - mode == 'times': {output_dir}/time{var}spectraKz.pdf
        """
        # 创建输出目录
        output_dir = 'spectra'
        os.makedirs(output_dir, exist_ok=True)

        if mode == 'slice':

            # 用一个字典将var转换为gmean
            spc_dict: Dict[str, Tuple[Spectrum, Spectrum]] = {
                'ME': (self.MEgmean, self.MEgstd),
                'KE': (self.KEgmean, self.KEgstd)
            }

            # 获取对应的平均能谱
            gmean, gstd = spc_dict[var]

            # 获取三个方向的能谱平均值与标准差, 并对另外两个方向的前三个数据点取平均
            # x方向: 保留x维度，对y和z的前3个点取平均
            gmean_x: np.ndarray = np.mean(np.mean(gmean.data[:, 0:3, 0:3], axis=2), axis=1)
            gstd_x : np.ndarray = np.mean(np.mean( gstd.data[:, 0:3, 0:3], axis=2), axis=1)

            # y方向: 保留y维度，对x和z的前3个点取平均
            gmean_y: np.ndarray = np.mean(np.mean(gmean.data[0:3, :, 0:3], axis=2), axis=0)
            gstd_y : np.ndarray = np.mean(np.mean( gstd.data[0:3, :, 0:3], axis=2), axis=0)

            # z方向: 保留z维度，对x和y的前3个点取平均
            gmean_z: np.ndarray = np.mean(np.mean(gmean.data[0:3, 0:3, :], axis=1), axis=0)
            gstd_z : np.ndarray = np.mean(np.mean( gstd.data[0:3, 0:3, :], axis=1), axis=0)

            # 对三个方向分别绘图
            for direction, k, gmean, gstd in [
                ('x', gmean.kx, gmean_x, gstd_x),
                ('y', gmean.ky, gmean_y, gstd_y),
                ('z', gmean.kz, gmean_z, gstd_z)
            ]:
                # 只取正半轴部分
                N = k.size
                k      : np.ndarray = k[1: N//2]
                gmean  : np.ndarray = gmean[1: N//2]
                gstd   : np.ndarray = gstd[1: N//2]

                # 创建新图
                plt.figure(figsize=(7, 5))

                # 绘制平均值曲线和标准差范围
                plt.loglog(k, gmean, 'k-', lw=2)
                plt.fill_between(k, gmean / gstd, gmean * gstd, color='gray', alpha=0.3, linewidth=0)

                # 设置标签和标题
                plt.xlabel(rf'$k_{direction}$', fontsize=12)
                plt.ylabel(rf'$E(k_{direction})$', fontsize=12)
                plt.title(self.title_dict[var], fontsize=14)

                # 添加网格
                plt.grid(True, which='both' , ls='--', lw=0.5, axis='x')  # x轴显示主次刻度
                plt.grid(True, which='major', ls='--', lw=0.5, axis='y')  # y轴只显示主刻度

                # 保存图像
                filename = f'avg{var}spectraK{direction}.pdf'
                plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
                plt.close()

        elif mode == 'times':

            # 用一个字典将var转换为spc
            spc_dict: Dict[str, List[Spectrum]] = {
                'ME': self.MEspectra,
                'KE': self.KEspectra
            }

            # 获取对应的能谱
            spectra: List[Spectrum] = spc_dict[var]
            times  : List[float]    = self.times

            # 设置颜色映射
            norm = Normalize(vmin=min(times), vmax=max(times))
            cmap = cm.get_cmap('jet')

            # 创建新图
            plt.figure(figsize=(10, 7))

            # 对每个时刻绘制能谱
            for spectrum, time in zip(spectra, times):
                # 获取z方向能谱（对xy方向积分）
                kz = spectrum.kz[1:spectrum.kz.size//2]  # 只取正半轴

                Ek_z = np.sum(np.sum(spectrum.data, axis=1), axis=0) * spectrum.dkx * spectrum.dky
                Ek_z = Ek_z[1:spectrum.kz.size//2] # 只取正半轴

                # 获取对应时间的颜色
                color = cmap(norm(time))

                # 绘制能谱
                plt.loglog(kz, Ek_z, color=color, linewidth=1)

            # 添加颜色条
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, pad=0.02)
            cbar.set_label('Time', fontsize=12)

            # 设置标签和标题
            plt.xlabel(r'$k_z$', fontsize=14)
            plt.ylabel(r'$E(k_z)$', fontsize=14)
            plt.title(self.title_dict[var], fontsize=14)

            # 添加网格
            plt.grid(True, which='both' , ls='--', lw=0.5, axis='x')  # x轴显示主次刻度
            plt.grid(True, which='major', ls='--', lw=0.5, axis='y')  # y轴只显示主刻度

            # 保存图像
            filename = f'time{var}spectraKz.pdf'
            plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
            plt.close()

    def plot(self) -> None:
        """绘制能谱: 打包实现所有功能
        """

        # 动能谱
        self.plot2d(var = 'ME')
        self.plot1d(var = 'ME', mode = 'slice')
        self.plot1d(var = 'ME', mode = 'times')
        
        # 磁能谱
        self.plot2d(var = 'KE')
        self.plot1d(var = 'KE', mode = 'slice')
        self.plot1d(var = 'KE', mode = 'times')
