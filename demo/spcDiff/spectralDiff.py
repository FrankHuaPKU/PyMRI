import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, cos, exp, pi, sin
import os

def spectralDiff(f, N, FP):
    """谱微分算法
    
    参数:
        f: 函数对象
        N: 采样点数
        FP: 浮点数精度 ('FP32' 或 'FP64')
    
    返回:
        x: 采样点数组
        fprime: 数值导数数组
    """
    # 设置数据类型
    if FP == 'FP32':
        dtype = np.float32
    elif FP == 'FP64':
        dtype = np.float64
    else:
        raise ValueError(f"未知精度: {FP}")
    
    # 在[-0.5, 0.5]区间上均匀采样
    x = np.linspace(-0.5, 0.5, N, endpoint=False, dtype=dtype)
    f_vals = f(x).astype(dtype)
    
    # 计算波数，注意规范化
    k = (2 * np.pi * np.fft.fftfreq(N, d=1.0/N)).astype(dtype)
    
    # 傅里叶变换
    f_hat = np.fft.fft(f_vals)
    
    # 频率空间中的导数
    fprime_hat = (1j * k * f_hat).astype(np.complex64 if FP == 'FP32' else np.complex128)

    # 计算截断位置
    # cutoff = int((1 - dealiasing) * N/2)
    # 去除中间的 dealiasing*N 个模式
    # fprime_hat[cutoff:-cutoff] = 0
    
    # 逆傅里叶变换，取实部
    fprime = np.real(np.fft.ifft(fprime_hat)).astype(dtype)
    
    return x, fprime

def centralDiff(f, N, FP):
    """中心差分算法
    
    参数:
        f: 函数对象
        N: 采样点数
        FP: 浮点数精度 ('FP32' 或 'FP64')
    
    返回:
        x: 采样点数组
        fprime: 数值导数数组
    """
    # 设置数据类型
    if FP == 'FP32':
        dtype = np.float32
    elif FP == 'FP64':
        dtype = np.float64
    else:
        raise ValueError(f"未知精度: {FP}")
    
    # 使用与谱方法相同的采样点
    x = np.linspace(-0.5, 0.5, N, endpoint=False, dtype=dtype)
    dx = x[1] - x[0]
    f_vals = f(x).astype(dtype)
    
    # 中心差分，考虑周期性边界条件
    fprime = np.zeros_like(f_vals, dtype=dtype)
    fprime[1:-1] = (f_vals[2:] - f_vals[:-2]) / (2 * dx)
    
    # 周期边界条件
    fprime[0] = (f_vals[1] - f_vals[-1]) / (2 * dx)
    fprime[-1] = (f_vals[0] - f_vals[-2]) / (2 * dx)
    
    return x, fprime

# 测试函数定义
def smooth(x):
    return np.cos(2 * np.pi * x) * np.exp(np.sin(2 * np.pi * x))

def multiscale(x):
    return np.sin(2*np.pi*x) + 0.8 * np.cos(4*np.pi*x + 0.1) + 0.3 * np.sin(16*np.pi*x - 0.2) \
        + 0.07 * np.cos(64*np.pi*x + 0.3) + 0.03 * np.exp(np.sin(128*np.pi*x + 0.9)) \
        + 0.01 * np.cos(256*np.pi*x - 0.1) + 0.007 * np.sin(512*np.pi*x - 0.2)

def nonperiodic(x):
    return (x+0.25)**2 * np.exp(-(x+0.25)**2)

def get_analytical_derivative(func_name):
    """获取解析导数
    
    参数:
        func_name: 函数名称
    
    返回:
        导数函数
    """
    x = symbols('x')
    function_dict = {
        'smooth'     : cos(2*pi*x) * exp(sin(2*pi*x)),
        'multiscale' : sin(2*pi*x) + 0.8*cos(4*pi*x + 0.1) + 0.3*sin(16*pi*x - 0.2) + 0.07*cos(64*pi*x + 0.3) \
                 + 0.03*exp(sin(128*pi*x + 0.9)) + 0.01*cos(256*pi*x - 0.1) + 0.007*sin(512*pi*x - 0.2),
        'nonperiodic': (x+0.25)**2 * exp(-(x+0.25)**2)
    }
    
    f = function_dict.get(func_name)
    if f is None:
        raise ValueError(f"未知函数: {func_name}")
    
    fprime = diff(f, x)
    return lambdify(x, fprime, modules=['numpy'])

def compute_error(numerical, analytical, x, func_name=''):
    """计算误差
    
    对于非周期函数，只计算x=0处的误差
    对于其他函数，计算L2范数误差
    """
    if func_name == 'nonperiodic':
        # 找到最接近x=0的点的索引
        center_idx = np.argmin(np.abs(x))
        # 计算x=0处的绝对误差
        return np.abs(numerical[center_idx] - analytical(x[center_idx]))
    else:
        # 其他函数使用L2范数误差
        return np.sqrt(np.mean((numerical - analytical(x))**2))

def plot_comparison(func, func_name, N, save_dir='outputs'):
    """绘制比较图并保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # 计算数值导数
    x_spec64, fprime_spec64 = spectralDiff(func, N, 'FP64')
    x_spec32, fprime_spec32 = spectralDiff(func, N, 'FP32')
    _, fprime_cd64 = centralDiff(func, N, 'FP64')
    _, fprime_cd32 = centralDiff(func, N, 'FP32')
    
    # 获取解析导数
    fprime_exact = get_analytical_derivative(func_name)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(x_spec64, fprime_exact(x_spec64), 'k-', label='Analytical', linewidth=2)
    plt.plot(x_spec64, fprime_spec64, 'r--', label='Spectral (FP64)', linewidth=2)
    plt.plot(x_spec32, fprime_spec32, 'r:', label='Spectral (FP32)', linewidth=2)
    plt.plot(x_spec64, fprime_cd64, 'b--', label='Central Diff (FP64)', linewidth=2)
    plt.plot(x_spec32, fprime_cd32, 'b:', label='Central Diff (FP32)', linewidth=2)
    
    plt.title(f'Derivative Comparison - {func_name.capitalize()} Function', pad=15)
    plt.xlabel('x', labelpad=10)
    plt.ylabel('f\'(x)', labelpad=10)
    plt.legend(frameon=True, edgecolor='black')
    plt.grid(True)
    
    # 如果是非周期函数，添加x=0处的标记
    if func_name == 'nonperiodic':
        center_idx = np.argmin(np.abs(x_spec64))
        plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        plt.plot(0, fprime_exact(x_spec64[center_idx]), 'ko', markersize=4, label='Error Evaluation Point')
        plt.legend()
    
    plt.savefig(os.path.join(save_dir, f'{func_name}_comparison.pdf'), bbox_inches='tight')
    plt.close()

def convergence_study(func, func_name, N_values, save_dir='outputs'):
    """收敛性研究"""
    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    errors_spec64 = []
    errors_spec32 = []
    errors_cd64 = []
    errors_cd32 = []
    
    fprime_exact = get_analytical_derivative(func_name)
    
    for N in N_values:
        x_spec64, fprime_spec64 = spectralDiff(func, N, 'FP64')
        x_spec32, fprime_spec32 = spectralDiff(func, N, 'FP32')
        _, fprime_cd64 = centralDiff(func, N, 'FP64')
        _, fprime_cd32 = centralDiff(func, N, 'FP32')
        
        err_spec64 = compute_error(fprime_spec64, fprime_exact, x_spec64, func_name)
        err_spec32 = compute_error(fprime_spec32, fprime_exact, x_spec32, func_name)
        err_cd64 = compute_error(fprime_cd64, fprime_exact, x_spec64, func_name)
        err_cd32 = compute_error(fprime_cd32, fprime_exact, x_spec32, func_name)
        
        errors_spec64.append(err_spec64)
        errors_spec32.append(err_spec32)
        errors_cd64.append(err_cd64)
        errors_cd32.append(err_cd32)

    
    # 绘制收敛性图
    plt.figure(figsize=(12, 6))
    plt.loglog(N_values, errors_spec64, 'rs-', label='Spectral (FP64)'    , linewidth=2, markersize=4)
    plt.loglog(N_values, errors_spec32, 'rs--' , label='Spectral (FP32)'    , linewidth=2, markersize=4)
    plt.loglog(N_values, errors_cd64  , 'bs-', label='Central Diff (FP64)', linewidth=2, markersize=4)
    plt.loglog(N_values, errors_cd32  , 'bs--' , label='Central Diff (FP32)', linewidth=2, markersize=4)
    
    plt.title(f'Error Convergence Analysis: {func_name.capitalize()} Function', pad=15)
    plt.xlabel('resolution (N)', labelpad=10)
    
    # 根据函数类型设置不同的y轴标签
    if func_name == 'nonperiodic':
        plt.ylabel('Absolute Error at x=0', labelpad=10)
    else:
        plt.ylabel('L2 Error', labelpad=10)
    
    plt.legend(frameon=True, edgecolor='black')
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f'{func_name}_convergence.pdf'), bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    # 测试函数列表
    test_functions = {
        'smooth': smooth,
        'multiscale': multiscale,
        'nonperiodic': nonperiodic
    }
    
    # 采样点数列表（用于收敛性研究）
    N_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    # 对每个测试函数进行测试
    for func_name, func in test_functions.items():
        print(f"正在计算测试函数: {func_name}")
        
        # 绘制导数比较图
        plot_comparison(func, func_name, N=1024)
        
        # 进行收敛性研究
        convergence_study(func, func_name, N_values)

if __name__ == '__main__':
    main()

    # print(np.linspace(-0.5, 0.5, 16, endpoint=False, dtype=np.float32))
    # print(np.linspace(-0.5, 0.5, 16, endpoint=True, dtype=np.float32))
