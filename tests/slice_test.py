import numpy as np
import sys
import os

# 添加src目录到Python路径
post_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/post'))
sys.path.insert(0, post_path)

import preprocess
from pymri import *

def test_slice():
    """测试切片图绘制功能
    
    在case目录中调用, 调用命令:
    python ../../../PyMRI/tests/slice_test.py
    """
    # 从输出文件中提取湍流场数据
    outn = 'prim'
    t1 = 50.0
    t2 = 55.0
    
    # 构建Turbulence对象
    turbulence = preprocess.output2turbulence(outn, t1, t2)
    
    # 绘制切片图
    plot2dslice(turbulence)

if __name__ == "__main__":
    test_slice()
