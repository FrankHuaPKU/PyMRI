import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, cos, exp, pi, sin
import os

def read_time_data(filename):
    """从数据文件中读取时间数据
    
    参数:
        filename: 数据文件路径
    
    返回:
        times: 时间数组
    """
    times = []
    with open(filename, 'r') as f:
        for line in f:
            # 提取time字段的值
            time = float(line.split('time=')[1].split()[0])
            times.append(time)
    return np.array(times)


def fornberg_weights(x, x0, M):
    """
    Compute finite difference weights for derivatives at x0,
    using Fornberg's recursive algorithm.

    符号尽可能参考Fornberg 1988论文的实现

    m: 导数阶数, 取值范围 0 ~ M
    n: 用于计算导数的数据点数量, 取值范围 0 ~ N (N + 1为所有数据点数量)
    i: 求和指标(文献中为nu), 取值范围 0 ~ n
    c[m, n, i]: 权重系数

    Parameters:
    ----------
    x : array_like
        Array of n+1 grid points (not necessarily uniformly spaced).
    x0 : float
        The point at which the derivative is to be approximated.
    M : int
        Maximum order of derivative required.

    Returns:
    -------
    c : ndarray of shape (n+1, M+1)
        Weights c[i, m] such that:
        f^(m)(x0) ≈ sum_i=0^n c[m, n, i] * f(x[i])
    """
    N = len(x) - 1
    c = np.zeros((M + 1, N + 1, N + 1))
    c[0, 0, 0] = 1.0
    c1 = 1.0

    for n in range(1, N + 1):
        c2 = 1.0
        for i in range(n):
            c3 = x[n] - x[i]
            c2 *= c3

            if n <= M: 
                c[n, n - 1, i] = 0

            for m in range(0, min(n, M) + 1):
                for j in range(n):
                    c[m, n, i] = ((x[n] - x0) * c[m, n - 1, i] - m * c[m - 1, n - 1, i]) / c3

        for m in range(0, min(n, M) + 1):
            c[m, n, n] = c1 / c2 * (m * c[m - 1, n - 1, n - 1] - (x[n - 1] - x0) * c[m, n - 1, n - 1])

        c1 = c2

    return c


def fornberg(f, times, t0, N):
    """使用Fornberg算法计算非均匀网格上的导数
    
    参数:
        f: 函数对象
        times: 时间点数组
        t0: 中心时刻
        N: 使用的数据点数
    
    返回:
        derivative: t0时刻的导数值
    """
    # 找到最接近t0的点的索引
    center_idx = np.argmin(np.abs(times - t0))
    half_N = N // 2
    
    # 提取N+1个点
    start_idx = max(0, center_idx - half_N)
    end_idx = min(len(times), center_idx + half_N + 1)
    
    # 如果在边界附近，调整索引以确保有足够的点
    if end_idx - start_idx < N + 1:
        if start_idx == 0:
            end_idx = N + 1
        else:
            start_idx = len(times) - (N + 1)
    
    # 提取时间点和函数值
    x = times[start_idx:end_idx]
    y = f(x)
    
    # 计算权重（只需要一阶导数，所以M=1）
    weights = fornberg_weights(x, t0, 1)
    
    # 计算导数（使用权重的第二列，即一阶导数的权重）
    derivative = np.sum(weights[1, N, :] * y)
    
    return derivative

def central_diff(f, times, t0, N):
    """使用中心差分计算导数（假设间隔近似均匀）
    
    参数:
        f: 函数对象
        times: 时间点数组
        t0: 中心时刻
        N: 使用的数据点数
    
    返回:
        derivative: t0时刻的导数值
    """
    # 找到最接近t0的点的索引
    center_idx = np.argmin(np.abs(times - t0))
    half_N = N // 2
    
    # 提取N+1个点
    start_idx = max(0, center_idx - half_N)
    end_idx = min(len(times), center_idx + half_N + 1)
    
    # 如果在边界附近，调整索引以确保有足够的点
    if end_idx - start_idx < N + 1:
        if start_idx == 0:
            end_idx = N + 1
        else:
            start_idx = len(times) - (N + 1)
    
    # 提取时间点和函数值
    x = times[start_idx:end_idx]
    y = f(x)
    
    # 计算平均间隔
    delta = (x[-1] - x[0]) / N
    
    # 根据N选择合适的差分格式
    if N == 2:
        # 二阶中心差分
        derivative = (y[2] - y[0]) / (2 * delta)
    elif N == 4:
        # 四阶中心差分
        derivative = (-y[4] + 8*y[3] - 8*y[1] + y[0]) / (12 * delta)
    elif N == 8:
        # 八阶中心差分
        derivative = (3*y[8] - 32*y[7] + 168*y[6] - 672*y[5] + 672*y[3] - 168*y[2] + 32*y[1] - 3*y[0]) / (840 * delta)
    elif N == 16:
        # 十六阶中心差分（使用简化的系数）
        c = np.array([1, -16, 120, -560, 1820, -4368, 8008, -11440, 0, 11440, -8008, 4368, -1820, 560, -120, 16, -1])
        derivative = np.sum(c * y) / (24024 * delta)
    elif N == 32:
        # 32 阶中心差分
        c = np.array([1, -32, 460, -4080, 25740, -118124, 421284, -1215280, 2874060, -5748120, 9914160, -14872800, 19831200, -23751200, 25581600, -24961440, 22314360, -18279000, 13572000, -8910960, 5103000, -2551500, 1062600, -354240, 90090, -15708, 1820, -120, 4, -1])
        derivative = np.sum(c * y) / (24024 * delta)

    else:
        raise ValueError(f"不支持的阶数: {N}")
    
    return derivative

# 测试函数定义
def smooth(x):
    """光滑测试函数"""
    return np.cos(2 * np.pi * x) * np.exp(np.sin(2 * np.pi * x))

def multiscale(x):
    """多尺度测试函数"""
    return np.sin(2*np.pi*x) + 0.8 * np.cos(4*np.pi*x + 0.1) + 0.3 * np.sin(16*np.pi*x - 0.2) \
        + 0.07 * np.cos(64*np.pi*x + 0.3) + 0.03 * np.exp(np.sin(128*np.pi*x + 0.9)) \
        + 0.01 * np.cos(256*np.pi*x - 0.1) + 0.007 * np.sin(512*np.pi*x - 0.2)

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
                 + 0.03*exp(sin(128*pi*x + 0.9)) + 0.01*cos(256*pi*x - 0.1) + 0.007*sin(512*pi*x - 0.2)
    }
    
    f = function_dict.get(func_name)
    if f is None:
        raise ValueError(f"未知函数: {func_name}")
    
    fprime = diff(f, x)
    return lambdify(x, fprime, modules=['numpy'])

def convergence_study(func, func_name, times, t0, N_values, save_dir='outputs'):
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
    
    errors_fornberg = []
    errors_central = []
    
    fprime_exact = get_analytical_derivative(func_name)
    exact_derivative = fprime_exact(t0)
    
    for N in N_values:
        # Fornberg方法
        fornberg_derivative = fornberg(func, times, t0, N)
        err_fornberg = abs(fornberg_derivative - exact_derivative)
        errors_fornberg.append(err_fornberg)
        
        # 中心差分方法（仅计算N≤16的情况）
        if N <= 16:
            try:
                central_derivative = central_diff(func, times, t0, N)
                err_central = abs(central_derivative - exact_derivative)
            except ValueError:
                err_central = np.nan
        else:
            err_central = np.nan
        errors_central.append(err_central)
    
    # 绘制收敛性图
    plt.figure(figsize=(12, 6))
    plt.loglog(N_values, errors_fornberg, 'rs-', label='Fornberg Method', linewidth=2, markersize=8)
    
    # 只绘制N≤16的中心差分结果
    valid_indices = [i for i, N in enumerate(N_values) if N <= 16]
    valid_errors = [errors_central[i] for i in valid_indices]
    plt.loglog(np.array(N_values)[valid_indices], valid_errors, 'bs-', 
              label='Central Difference', linewidth=2, markersize=8)
    
    plt.title(f'Error Convergence Analysis: {func_name.capitalize()} Function', pad=15)
    plt.xlabel('Order (N)', labelpad=10)
    plt.ylabel('Absolute Error', labelpad=10)

    plt.xticks(N_values, [str(N) for N in N_values])  # 只显示指定的N
    
    plt.legend(frameon=True, edgecolor='black')
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f'{func_name}_convergence.pdf'), bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 读取时间数据
    times = read_time_data('data.txt')
    
    # 获取中心时刻
    t0 = times[len(times)//2]
    
    # 测试函数列表
    test_functions = {
        'smooth': smooth,
        'multiscale': multiscale
    }
    
    # 使用的数据点数列表
    N_values = [2, 4, 8, 16, 32]
    
    # 对每个测试函数进行测试
    for func_name, func in test_functions.items():
        print(f"正在计算测试函数: {func_name}")
        convergence_study(func, func_name, times, t0, N_values)

if __name__ == '__main__':
    main()
