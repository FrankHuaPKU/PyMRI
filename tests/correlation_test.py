import numpy as np

from pymri import *

import sys
import os

# 添加src目录到Python路径
post_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/post'))
sys.path.insert(0, post_path)

import preprocess


def test_Correlation():
    """测试Correlation类
    
    在case目录中调用, 调用命令:
    python ../../../PyMRI/tests/correlation_test.py
    """
    # 从输出文件中提取湍流场数据
    outn = 'prim'
    t1 = 125.0
    t2 = 200.0
    
    # 构建Turbulence对象
    turbulence = preprocess.output2turbulence(outn, t1, t2)
        
    # 计算关联函数
    corr = Correlation(turbulence, normalized=True)
    
    # 绘制关联函数
    corr.plot()


if __name__ == "__main__":
    test_Correlation()
