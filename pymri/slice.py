import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from typing import Dict, Tuple, List
import os

from .turbulence import Turbulence

# 获取字体文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(current_dir, '../tools/fonts/Times New Roman.ttf')

# 添加字体文件
font_files = fm.fontManager.addfont(font_path)
font = fm.FontProperties(fname=font_path)
# 获取字体的精确名称
TimesNewRoman = font.get_name()

# 设置 matplotlib 参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = TimesNewRoman
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

# 设置全局字体大小
plt.rcParams['font.size'] = 18  # 默认字体大小
plt.rcParams['axes.labelsize'] = 18  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 18  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 16  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 16  # y轴刻度标签字体大小
plt.rcParams['figure.titlesize'] = 16  # 图形标题字体大小
  

def get_cmap_range(turbulence: Turbulence) -> Dict[str, Tuple[float, float]]:
    """计算不同物理量的colormap范围
    
    参数:
        turbulence: Turbulence对象, 包含所有物理量数据
        
    返回值:
        包含各物理量colormap范围的字典:
        {
            'rho': (vmin, vmax),  # 密度场范围
            'vel': (vmin, vmax),  # 速度场范围
            'B'  : (vmin, vmax)   # 磁场范围
        }
    """
    ranges = {}
    
    # 计算密度场范围
    # 将所有时间切片的数据展平并连接
    rho_data = np.concatenate([rho.data.flatten() for rho in turbulence.rhos])
    rho_mean = np.mean(rho_data)
    rho_delta = max(np.max(rho_data) - rho_mean, rho_mean - np.min(rho_data))
    ranges['rho'] = (rho_mean - rho_delta, rho_mean + rho_delta)
    
    # 计算速度场范围
    # 将所有时间切片的三个分量数据展平并连接
    vel_data = np.concatenate([
        np.concatenate([V.x.flatten(), V.y.flatten(), V.z.flatten()])
        for V in turbulence.Vs
    ])
    vel_max = np.max(np.abs(vel_data))
    ranges['vel'] = (-vel_max, vel_max)
    
    # 计算磁场范围
    # 将所有时间切片的三个分量数据展平并连接
    B_data = np.concatenate([
        np.concatenate([B.x.flatten(), B.y.flatten(), B.z.flatten()])
        for B in turbulence.Bs
    ])
    B_max = np.max(np.abs(B_data))
    ranges['B'] = (-B_max, B_max)
    
    return ranges


def plot2dslice(turbulence: Turbulence) -> None:
    """绘制所有物理量的三个方向切片图
    
    参数:
        turbulence: Turbulence对象，包含所有物理量数据
        
    每个物理量分量生成一张图，包含三个方向的切片:
    - 左上: yx平面 (z=0)
    - 右上: zx平面 (y=0)
    - 左下: yz平面 (x=0)
    
    所有图片按照如下目录结构保存:
    slice/
    ├── rho/
    ├── Vx/
    ├── Vy/
    ├── Vz/
    ├── Bx/
    ├── By/
    └── Bz/
    """
    # 获取colormap范围
    cmap_ranges: Dict[str, Tuple[float, float]] = get_cmap_range(turbulence)
    
    # 创建主目录
    base_dir = 'slice'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 遍历所有时间切片
    for time_index, time in enumerate(turbulence.times):
        # 获取当前时间切片的数据
        rho = turbulence.rhos[time_index]
        V   = turbulence.Vs[time_index]
        B   = turbulence.Bs[time_index]
        
        # 获取网格尺寸并计算中心切片位置
        Nx, Ny, Nz = rho.data.shape
        mid_x = Nx // 2
        mid_y = Ny // 2
        mid_z = Nz // 2
        
        # 定义要绘制的物理量配置
        field_configs: List[Tuple[str, List[Tuple[str, np.ndarray]], str]] = [
            ('density' , [('rho', rho.data)], 'rho'),
            ('velocity', [('Vx', V.x), ('Vy', V.y), ('Vz', V.z)], 'vel'),
            ('magnetic', [('Bx', B.x), ('By', B.y), ('Bz', B.z)], 'B')
        ]
        
        # 遍历每个物理量
        for field_name, var, range_key in field_configs:
            # 遍历物理量的每个分量
            for var_name, var_data in var:
                # 创建分量对应的子目录
                var_dir = os.path.join(base_dir, var_name)

                if not os.path.exists(var_dir):
                    os.makedirs(var_dir)
                
                # 创建带有特定尺寸比例的图形
                fig = plt.figure(figsize=(14, 8), constrained_layout=False)
                
                # 创建网格布局，确保im1和im2的高度相同
                # 第一行占据2/3高度，第二行占据1/3高度
                # 第一列占据2/3宽度，第二列占据1/3宽度
                gs = plt.GridSpec(2, 3, figure=fig,
                                width_ratios=[4, 1, 0.2],  # 最后一列宽度为0.2，专门放colorbar
                                height_ratios=[2, 1],
                                left=0.1,    # 左边距
                                right=0.9,   # 右边距
                                top=0.9,     # 上边距
                                bottom=0.1,  # 下边距
                                wspace=0.08, # 水平间距
                                hspace=0.08) # 垂直间距
                
                # 创建三个子图并指定位置
                ax1 = fig.add_subplot(gs[0, 0])    # yx平面
                ax2 = fig.add_subplot(gs[0, 1])    # zx平面
                ax3 = fig.add_subplot(gs[1, 0])    # yz平面
                
                # colorbar专用子图
                cax = fig.add_subplot(gs[:, 2])  # 占用第3列的所有行
                
                # 获取对应的colormap范围
                vmin, vmax = cmap_ranges[range_key]
                
                # 获取物理坐标范围
                Lx, Ly, Lz = rho.box
                x = np.linspace(-Lx/2, Lx/2, Nx)
                y = np.linspace(-Ly/2, Ly/2, Ny)
                z = np.linspace(-Lz/2, Lz/2, Nz)
                
                # 绘制切片图
                # 注意: imshow 绘制数组的第一个维度对应纵轴
                # 设置colormap
                if var_name == 'rho':
                    cmap = 'RdBu'
                else:
                    cmap = 'RdBu'
                
                im1 = ax1.imshow(var_data[:, :, mid_z]  , origin='upper', 
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               extent=[-Ly/2, Ly/2, Lx/2, -Lx/2])
                im2 = ax2.imshow(var_data[:, mid_y, :]  , origin='upper',
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               extent=[-Lz/2, Lz/2, Lx/2, -Lx/2])
                im3 = ax3.imshow(var_data[mid_x, :, :].T, origin='lower',
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               extent=[-Ly/2, Ly/2, -Lz/2, Lz/2])
                
                # 设置纵横比
                ax1.set_aspect('equal')
                ax2.set_aspect('equal')
                ax3.set_aspect('equal')
                
                # 调整刻度标签位置
                # im1只保留左轴刻度标签
                ax1.xaxis.set_ticklabels([])  # 去掉上轴刻度标签
                
                # im2只保留下轴刻度标签，去掉纵轴刻度标签和标签
                ax2.yaxis.set_ticklabels([])
                ax2.set_ylabel('')  # 去掉纵轴标签
                
                # im3保持默认的下轴和左轴刻度标签
                
                # 设置所有刻度线朝向图片内部
                ax1.tick_params(direction='in')
                ax2.tick_params(direction='in')
                ax3.tick_params(direction='in')
                
                # 添加坐标轴标签
                # 统一调整字体大小
                ax1.set_ylabel(r'$x/H$', labelpad=0)

                ax2.set_xlabel(r'$z/H$', labelpad=10)

                ax3.set_xlabel(r'$y/H$', labelpad=10)
                ax3.set_ylabel(r'$z/H$', labelpad=0)
  
                # 设置刻度、边框线宽
                linewidth = 1.5  

                # 设置刻度数字到轴线的距离
                tick_pad = 7 

                ax1.tick_params(width=linewidth, pad=tick_pad)
                ax2.tick_params(width=linewidth, pad=tick_pad)
                ax3.tick_params(width=linewidth, pad=tick_pad)
                
                # 调整边框线条粗细
                for ax in [ax1, ax2, ax3]:
                    for spine in ax.spines.values():
                        spine.set_linewidth(linewidth)
                
                # 添加colorbar到专用子图
                cbar = fig.colorbar(im1, cax=cax, shrink=0.9)  # 设置colorbar高度为原始高度的90%

                # 设置colorbar标签, 用字典映射变量名
                varname_dict = {
                    'rho': r'$\rho$',
                    'Vx' : r'$V_x$',
                    'Vy' : r'$V_y$',
                    'Vz' : r'$V_z$',
                    'Bx' : r'$B_x$',
                    'By' : r'$B_y$',
                    'Bz' : r'$B_z$'
                }
                
                cbar.set_label(f'{varname_dict[var_name]}', labelpad=10) 
                # cbar.ax.tick_params(labelsize=16)
                
                # 设置colorbar边框
                cbar.outline.set_linewidth(linewidth)  # 设置colorbar边框粗细
                
                # 保存图片到对应子目录
                plt.savefig(os.path.join(var_dir, f't={time:.1f}.pdf'), bbox_inches='tight')
                plt.close()
