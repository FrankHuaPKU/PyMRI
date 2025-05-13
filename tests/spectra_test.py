import numpy as np

from pymri import *

import sys
import os

# 添加src目录到Python路径
post_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/post'))
sys.path.insert(0, post_path)

import preprocess

def test_spectrum():
    """测试Spectrum类"""
    # 创建测试数据
    data = np.array([[[0.98, 1.04],
                      [0.96, 1.02]]])
    box = [2.0, 3.0, 1.0]
    
    # 创建Spectrum实例
    spc = Spectrum(data, box)

    spc = spc + spc
    
    # 测试基本属性
    print(spc.data)


def test_MagneticSpectra():
    """测试MagneticSpectra类
    
    在case目录中调用, 调用命令:
    srun -N 1 -n 1 -J hyy python ../../../PyMRI/tests/spectra_test.py
    """
    # 从输出文件中提取湍流场数据
    outn = 'prim'
    t1 = 50.0
    t2 = 100.0
    
    # 构建Turbulence对象
    turbulence = preprocess.output2turbulence(outn, t1, t2)
        
    # 计算磁场能谱
    spc = MagneticSpectra(turbulence)
    
    # 绘制能谱
    spc.plot()


if __name__ == "__main__":
    test_MagneticSpectra()