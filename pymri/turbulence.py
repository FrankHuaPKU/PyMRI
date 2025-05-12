import numpy as np
from typing import List, Tuple, Union, Optional

class ScalarField:
    """标量场类
    
    用于表示密度等标量物理量场。
    
    属性:
        data (np.ndarray) : 标量场数据
        Lx, Ly, Lz (float): 模拟box的x, y, z方向尺寸
        Nx, Ny, Nz (int)  : 模拟box的x, y, z方向网格分辨率
        dx, dy, dz (float): 单个网格的x, y, z方向尺寸
        dxdydz (float)    : 体积元

    支持的运算:
        1. 各类NumPy的通用函数(如np.sqrt)  np.sqrt(ScalarField) -> ScalarField
        2. 加减: ScalarField + ScalarField -> ScalarField
        3. 乘法: ScalarField * ScalarField -> ScalarField
        4. 数乘: ScalarField * float -> ScalarField, 
                float * ScalarField -> ScalarField
        5. 除法: ScalarField / ScalarField -> ScalarField, 
                ScalarField / float -> ScalarField, 
                float / ScalarField -> ScalarField
        6. 幂运算: ScalarField ** float -> ScalarField

    注意: 使用type(self)表示当前类型, 一般即为ScalarField类本身, 但被继承时则表示子类
    """
    
    def __init__(self, data: np.ndarray, box: List[float]):
        """初始化标量场
        
        参数:
            data: 标量场数据
            box: [Lx, Ly, Lz] 模拟box的三个方向尺寸
        """
        self.data = data
        self.box  = box
        self.Lx, self.Ly, self.Lz = box

        # 计算网格分辨率与网格尺寸
        self.Nx, self.Ny, self.Nz = data.shape
        self.dx, self.dy, self.dz = self.Lx/self.Nx, self.Ly/self.Ny, self.Lz/self.Nz
        self.dxdydz = self.dx * self.dy * self.dz
        
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """支持NumPy的通用函数操作, 例如np.sqrt等等"""
        # 处理输入参数
        processed_inputs = []
        for input in inputs:
            if isinstance(input, type(self)):
                processed_inputs.append(input.data)
            else:
                processed_inputs.append(input)
                
        # 执行ufunc操作
        results = getattr(ufunc, method)(*processed_inputs, **kwargs)
        
        # 处理返回值
        if isinstance(results, tuple):
            # 多返回值情况
            return tuple(
                type(self)(result, [self.Lx, self.Ly, self.Lz]) 
                if isinstance(result, np.ndarray) else result 
                for result in results
            )
        elif isinstance(results, np.ndarray):
            # 返回数组, 转换为当前类型
            return type(self)(results, [self.Lx, self.Ly, self.Lz])
        else:
            # 返回标量值
            return results
            
    def __add__(self, other):
        """加法运算
        
        要求other必须是同一类型
        
        调用示例:
            field1 = ScalarField(data1, box)
            field2 = ScalarField(data2, box)
            field = field1 + field2
        """
        if not isinstance(other, type(self)):
            raise TypeError("只有同一类型的对象才能相加")
            
        if (self.box != other.box):
            raise ValueError("只有box尺寸相同的两个对象才能相加")
            
        if self.data.shape != other.data.shape:
            raise ValueError("两个对象的数组形状必须相同")

        return type(self)(self.data + other.data, [self.Lx, self.Ly, self.Lz])

    def __neg__(self):
        """负数运算
        
        实现 -ScalarField 操作，返回每个分量取反的标量场
        
        调用示例：  
            rho = ScalarField(data, box)
            negative_rho = -rho  # 返回 ScalarField(-data, box)
        """
        return type(self)(-self.data, [self.Lx, self.Ly, self.Lz])
            
    def __sub__(self, other):
        """减法运算
        
        要求other必须是ScalarField类型
        
        调用示例:
            field1 = ScalarField(data1, box)
            field2 = ScalarField(data2, box)
            field = field1 - field2
        """
        if not isinstance(other, type(self)):
            raise TypeError("只有同一类型的对象才能相减")
            
        if (self.box != other.box):
            raise ValueError("只有box尺寸相同的两个对象才能相减")
            
        if self.data.shape != other.data.shape:
            raise ValueError("两个对象的数组形状必须相同")

        return type(self)(self.data - other.data, [self.Lx, self.Ly, self.Lz])
            
    def __mul__(self, other):
        """乘法运算
        
        支持运算:
        1. 与浮点数或ScalarField相乘
        2. 与VectorField相乘时, 委托给VectorField的__rmul__方法处理

        调用示例:
            field1 = ScalarField(data1, box)
            field2 = ScalarField(data2, box)
            field = field1 * field2
        """
        if isinstance(other, (float, int)):
            return type(self)(self.data * other, [self.Lx, self.Ly, self.Lz])
        elif isinstance(other, type(self)):
            return type(self)(self.data * other.data, [self.Lx, self.Ly, self.Lz])
        elif isinstance(other, VectorField):
            # 委托给VectorField处理
            return other.__rmul__(self)
        else:
            raise TypeError("ScalarField只能与浮点数、ScalarField或VectorField相乘")
            
    def __rmul__(self, other):
        """右乘法运算
        
        支持:
        1. 浮点数 * ScalarField
        2. VectorField * ScalarField (委托给VectorField的__mul__方法)

        调用示例:
            field1 = ScalarField(data1, box)
            field2 = ScalarField(data2, box)
            field = field1 * field2
        """
        if isinstance(other, (float, int)):
            return type(self)(other * self.data, [self.Lx, self.Ly, self.Lz])
        elif isinstance(other, VectorField):
            # 委托给VectorField处理
            return other.__mul__(self)
        else:
            raise TypeError("ScalarField只能与浮点数、ScalarField或VectorField相乘")

    def __truediv__(self, other):
        """除法运算
        支持运算:
        1. 与浮点数或整数相除
        2. 与ScalarField相除时, 委托给ScalarField的__truediv__方法处理
        
        调用示例:
            field1 = ScalarField(data1, box)
            field2 = ScalarField(data2, box)
            field = field1 / field2
        """
        if isinstance(other, (float, int)):
            return type(self)(self.data / other, [self.Lx, self.Ly, self.Lz])
        elif isinstance(other, type(self)):
            return type(self)(self.data / other.data, [self.Lx, self.Ly, self.Lz])
        else:
            raise TypeError("ScalarField只能与浮点数、整数或ScalarField相除")
            
    def __rtruediv__(self, other):
        """右除法运算

        支持运算:
        1. 浮点数 / ScalarField
        
        调用示例:
            field1 = ScalarField(data1, box)
            field2 = ScalarField(data2, box)
            field = field1 / field2
        """
        if isinstance(other, (float, int)):
            return type(self)(other / self.data, [self.Lx, self.Ly, self.Lz])
        else:
            raise TypeError("只支持浮点数除以ScalarField")

    def __pow__(self, other):
        """幂运算

        支持运算:
        1. 标量幂运算: ScalarField ** float -> ScalarField

        调用示例:
            field = ScalarField(data, box)
            field_squared = field ** 2
        """
        if isinstance(other, (float, int)):
            return type(self)(self.data ** other, [self.Lx, self.Ly, self.Lz])
        else:
            raise TypeError("只支持标量幂运算")
            

    @property
    def mean(self) -> float:
        """计算一个标量场的平均值
        调用示例:
            field = ScalarField(data, box)
            field_mean = field.mean # 计算标量场的空间平均值
        """
        return np.mean(self.data)
    
    @property
    def std(self) -> float:
        """计算一个标量场的标准差
        调用示例:
            field = ScalarField(data, box)
            field_std = field.std # 计算标量场的空间标准差
        """
        return np.std(self.data)

    @property
    def total(self) -> float:
        """计算一个标量场的总量，例如密度场的总质量
        调用示例:
            field = ScalarField(data, box)
            field_total = field.total # 计算标量场的空间总和
        """
        return np.sum(self.data) * self.dxdydz
    




class VectorField:
    """矢量场类
    
    用于表示速度场、磁场等矢量物理量场。
    
    属性:
        x, y, z (np.ndarray): x, y, z分量
        norm (np.ndarray): 矢量场的模
        Lx, Ly, Lz (float): 模拟box的x, y, z方向尺寸
        Nx, Ny, Nz (int)  : 模拟box的x, y, z方向网格分辨率
        dx, dy, dz (float): 模拟box的x, y, z方向网格尺寸
        dxdydz (float)    : 体积元

    支持的运算:
        1. 加减: VectorField + VectorField -> VectorField, 
                VectorField - VectorField -> VectorField
        2. 数乘: VectorField * ScalarField -> VectorField, 
                ScalarField * VectorField -> VectorField,
                VectorField * float -> VectorField, 
                float * VectorField -> VectorField
        3. 点乘: VectorField * VectorField -> ScalarField
        4. 叉乘: VectorField ** VectorField -> VectorField
        5. 除法: VectorField / ScalarField -> VectorField, 
                VectorField / float -> VectorField, 
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, box: List[float]):
        """初始化矢量场
        
        参数:
            x: x分量数据
            y: y分量数据
            z: z分量数据
            box: [Lx, Ly, Lz] 模拟box的三个方向尺寸

            dxdydz: 单个网格的体积
        """
        self.x = x
        self.y = y
        self.z = z
        self.box = box
        self.Lx, self.Ly, self.Lz = box

        # 计算网格分辨率与网格尺寸
        self.Nx, self.Ny, self.Nz = x.shape
        self.dx, self.dy, self.dz = self.Lx/self.Nx, self.Ly/self.Ny, self.Lz/self.Nz
        self.dxdydz = self.dx * self.dy * self.dz

    @property
    def norm(self) -> 'ScalarField':
        """计算矢量场的模, 返回一个标量场
        
        调用示例:
            field = VectorField(x, y, z, box)
            field_norm = field.norm # 计算矢量场的模
        """
        return ScalarField(np.sqrt(self.x**2 + self.y**2 + self.z**2), self.box)

    def __add__(self, other):
        """加法运算
        
        支持:
        1. 加法: VectorField + VectorField -> VectorField
        """
        if isinstance(other, VectorField):
            return VectorField(self.x + other.x, self.y + other.y, self.z + other.z, self.box)
        else:
            raise TypeError("VectorField只能与另一个VectorField相加")

    def __neg__(self):
        """相反矢量
        
        实现 -VectorField 操作，返回每个分量取反的矢量场
        
        调用示例：
            V = VectorField(x, y, z, box)
            negative_V = -V  # 返回 VectorField(-x, -y, -z, box)
        """
        return VectorField(-self.x, -self.y, -self.z, self.box)

    def __sub__(self, other):
        """矢量减法

        利用相反矢量实现
        
        支持:
        1. 减法: VectorField - VectorField -> VectorField
        """
        if isinstance(other, VectorField):
            return self + (-other)
        else:
            raise TypeError("VectorField只能与另一个VectorField相减")
        
    def __mul__(self, other):
        """乘法运算
        
        支持:
        1. 数乘: VectorField * ScalarField -> VectorField
        2. 点乘: VectorField * VectorField -> ScalarField
        3. 系数: VectorField * float -> VectorField
        """
        if isinstance(other, VectorField):
            # 点乘
            return ScalarField(
                self.x * other.x + self.y * other.y + self.z * other.z,
                [self.Lx, self.Ly, self.Lz]
            )
        elif isinstance(other, ScalarField):
            # 数乘
            return VectorField(
                self.x * other.data,
                self.y * other.data,
                self.z * other.data,
                [self.Lx, self.Ly, self.Lz]
            )
        elif isinstance(other, (float, int)):
            # 数乘（标量）
            return VectorField(
                self.x * other,
                self.y * other,
                self.z * other,
                [self.Lx, self.Ly, self.Lz]
            )
        else:
            # 返回NotImplemented让Python尝试其他方法
            return NotImplemented
            
    def __rmul__(self, other):
        """右乘法运算
        
        支持:
        1. 数乘: ScalarField * VectorField -> VectorField
        2. 系数: float * VectorField -> VectorField
        
        注意: VectorField * VectorField是点乘，不需要在这里处理，
        因为左操作数是VectorField时会调用__mul__
        """
        if isinstance(other, ScalarField):
            # 数乘
            return VectorField(
                other.data * self.x,
                other.data * self.y,
                other.data * self.z,
                [self.Lx, self.Ly, self.Lz]
            )
        elif isinstance(other, (float, int)):
            # 数乘（标量）
            return VectorField(
                other * self.x,
                other * self.y,
                other * self.z,
                [self.Lx, self.Ly, self.Lz]
            )
        else:
            return NotImplemented
        
    def __pow__(self, other):
        """叉乘运算: 
        VectorField ** VectorField -> VectorField
        """
        if isinstance(other, VectorField):
            return VectorField(
                self.y * other.z - self.z * other.y,
                self.z * other.x - self.x * other.z,
                self.x * other.y - self.y * other.x,
                [self.Lx, self.Ly, self.Lz]
            )
        else:
            raise TypeError("叉乘运算仅支持两个VectorField相乘")




def sqrt(field: 'ScalarField') -> 'ScalarField':
    """计算标量场的平方根
    
    调用示例:
        field = ScalarField(data, box)
        sqrt_field = sqrt(field)
    """

    # 如果field是ScalarField类型, 则计算其平方根; 否则报错
    if isinstance(field, ScalarField):
        return ScalarField(np.sqrt(field.data), [field.Lx, field.Ly, field.Lz])
    else:
        raise TypeError("sqrt()仅支持ScalarField类型参数")


# 计算散度, 梯度, 旋度
# 使用FFT计算所有微分


def grad(field: 'ScalarField') -> 'VectorField':
    """计算标量场的梯度
    
    调用示例:
        field = ScalarField(data, box)
        grad_field = grad(field)
    """
    pass


def div(field: 'VectorField') -> 'ScalarField':
    """计算矢量场的散度
    
    调用示例:
        field = VectorField(x, y, z, box)
        div_field = div(field)
    """
    pass


def curl(field: 'VectorField') -> 'VectorField':
    """计算矢量场的旋度
    
    调用示例:
        field = VectorField(x, y, z, box)
        curl_field = curl(field)
    """
    pass



def avg(fields: Union[List[ScalarField], List[VectorField]]) -> Union[ScalarField, VectorField]:
    """计算多个场的平均场
    
    参数:
        fields: 标量场列表或矢量场列表
        
    返回:
        平均场（标量场或矢量场）
        
    示例:
        fields = [ScalarField(data1, box), ScalarField(data2, box), ...]
        avg_field = avg(fields)  # 返回平均标量场
        
    """
    if not fields:
        raise ValueError("输入的场列表不能为空")
        
    # 获取第一个场的类型和box尺寸
    first_field = fields[0]
    box = [first_field.Lx, first_field.Ly, first_field.Lz]
    
    if isinstance(first_field, ScalarField):
        # 处理标量场列表
        # 将所有场的data堆叠成一个三维数组，然后沿第一维取平均
        data_stack = np.stack([field.data for field in fields])
        avg_data = np.mean(data_stack, axis=0)
        return ScalarField(avg_data, box)
        
    elif isinstance(first_field, VectorField):
        # 处理矢量场列表
        # 分别对x、y、z分量进行平均
        x_stack = np.stack([field.x for field in fields])
        y_stack = np.stack([field.y for field in fields])
        z_stack = np.stack([field.z for field in fields])
        
        avg_x = np.mean(x_stack, axis=0)
        avg_y = np.mean(y_stack, axis=0)
        avg_z = np.mean(z_stack, axis=0)
        
        return VectorField(avg_x, avg_y, avg_z, box)
    else:
        raise TypeError("不支持的场类型, 必须是ScalarField或VectorField")



def std(fields: Union[List[ScalarField], List[VectorField]]) -> Union[ScalarField, VectorField]:
    """计算多个场的标准差场
    
    参数:
        fields: 标量场列表或矢量场列表
        
    返回:
        标准差场（标量场或矢量场）
        
    示例:
        fields = [ScalarField(data1, box), ScalarField(data2, box), ...]
        std_field = std(fields)  # 返回标准差标量场
    """
    if not fields:
        raise ValueError("输入的场列表不能为空")
        
    # 获取第一个场的类型和box尺寸
    first_field = fields[0]
    box = [first_field.Lx, first_field.Ly, first_field.Lz]
    
    if isinstance(first_field, ScalarField):
        # 处理标量场列表
        # 将所有场的data堆叠成一个三维数组，然后沿第一维取标准差
        data_stack = np.stack([field.data for field in fields])
        std_data = np.std(data_stack, axis=0)
        return ScalarField(std_data, box)
        
    elif isinstance(first_field, VectorField):
        # 处理矢量场列表
        # 分别对x、y、z分量计算标准差
        x_stack = np.stack([field.x for field in fields])
        y_stack = np.stack([field.y for field in fields])
        z_stack = np.stack([field.z for field in fields])
        
        std_x = np.std(x_stack, axis=0)
        std_y = np.std(y_stack, axis=0)
        std_z = np.std(z_stack, axis=0)
        
        return VectorField(std_x, std_y, std_z, box)
    else:
        raise TypeError("不支持的场类型, 必须是ScalarField或VectorField")






class Turbulence:
    """MRI湍流场类
    
    用于存储和处理MRI湍流相关的物理量场。
    
    属性:
        case (str)              : 湍流对应的case名
        rhos (List[ScalarField]): 密度场列表
        Vs (List[VectorField])  : 速度场列表
        Bs (List[VectorField])  : 磁场列表
        times (List[float])     : 对应的模拟时刻列表
        q (float)               : 剪切参量
        EoS (str)               : 状态方程, 可选'isothermal'/'adiabatic'/'incompressible'

        nu (float)              : 粘度
        eta (float)             : 磁耗散系数(电阻率)

        Pm (float)              : 磁Prandtl数, Pm = nu / eta

        wVs (List[VectorField]) : 密度加权速度场列表
        avgBs (List[Tuple[float, float, float]]) : 平均磁场
        KEs (List[float])       : 总动能
        MEs (List[float])       : 总磁能
    """
    
    def __init__(
        self,
        case : str,
        rhos : List[ScalarField],
        Vs   : List[VectorField],
        Bs   : List[VectorField],
        times: List[float],
        q    : float,
        EoS  : str,
        nu   : float,
        eta  : float
    ):
        """初始化湍流场"""

        self.case  = case  # case名
        self.rhos  = rhos  # 密度场列表
        self.Vs    = Vs    # 速度场列表
        self.Bs    = Bs    # 磁场列表
        self.times = times # 对应的模拟时刻列表
        self.q     = q     # 剪切参量
        self.EoS   = EoS   # 状态方程类型
        
        self.nu    = nu    # 粘度
        self.eta   = eta   # 磁耗散系数(电阻率)
        self.Pm    = nu / eta # 磁Prandtl数
        # 验证输入数据的一致性
        n_snapshots = len(times)
        if not all(len(x) == n_snapshots for x in [rhos, Vs, Bs]):
            raise ValueError("物理量场列表长度与时间列表长度不一致")
            
        if EoS not in ['isothermal', 'adiabatic', 'incompressible']:
            raise ValueError("不支持的状态方程类型")

            
    @property
    def wVs(self) -> Optional[List[VectorField]]:
        """计算加权速度场 sqrt(rho) * V
        
        仅在EoS != 'incompressible'时有效
        """
        if self.EoS == 'incompressible':
            return self.Vs

        # 简洁实现: 利用 ScalarField 与 VectorField 的自定义运算
        return [sqrt(rho) * V for rho, V in zip(self.rhos, self.Vs)]

        
    @property
    def avgBs(self) -> List[Tuple[float, float, float]]:
        """计算平均磁场
        
        调用:
            turbulence.avgBs: 三元组列表，对应每个时刻的平均磁场三分量
        
        """
        return [ ( np.mean(B.x), np.mean(B.y), np.mean(B.z) ) for B in self.Bs ]
        

    @property
    def KEs(self) -> List[float]:
        """计算总动能
        
        调用:
            turbulence.KEs: 列表，对应每个时刻的总动能
        """
        return [ 0.5 * (wV.norm**2).total for wV in self.wVs ]
        

    @property
    def MEs(self) -> List[float]:
        """计算总磁能
        
        调用:
            turbulence.MEs: 列表，对应每个时刻的总磁能
        """
        return [ 0.5 * (B.norm**2).total  for B  in self.Bs  ]
    

    @property
    def density_fluctuations(self) -> List[float]:
        """ 计算每个时刻的相对密度涨落 δρ/ρ 
        
        调用:
            turbulence.density_fluctuations: 列表，对应每个时刻的相对密度涨落 δρ/ρ
        """
        # 直接使用ScalarField对象的属性方法.std和.mean计算
        return [rho.std / rho.mean for rho in self.rhos]



def tests():
    """测试ScalarField, VectorField, Turbulence三个数据类型"""

    # 测试ScalarField
    # 创建形状为 (2,2,1) 的三维数组
    # 使用 reshape 来调整维度顺序
    data = np.array([[[0.98, 1.04],
                      [0.96, 1.02]]])
    box = [2.0, 3.0, 1.0]
    rho = ScalarField(data, box)

    '''
    print(rho.data)

    print(rho.data)
    print(rho.mean)
    print(rho.std)
    print(rho.total)
    ''' 

    # 测试VectorField
    V = VectorField(np.ones_like(data), np.ones_like(data), np.zeros_like(data), box)

    '''
    print(V.x)
    print(V.y)
    print(V.z)
    print(V.norm.data)
    '''

    B = VectorField(np.zeros_like(data), np.zeros_like(data), np.ones_like(data), box)

    '''
    print((-V * B).data)
    print((rho * V).x)
    print(((V + B) ** (V - B)).x)
    '''

    # 测试Turbulence
    turbulence = Turbulence(
        case  = 'test',
        rhos  = [rho],
        Vs    = [V],
        Bs    = [B],
        times = [0.0],
        q     = 1.5,
        EoS   = 'isothermal',
        nu    = 0.0001,
        eta   = 0.00001
    )

    print((turbulence.wVs)[0].x)
    print(turbulence.avgBs)
    print(turbulence.KEs)
    print(turbulence.MEs)
    print(turbulence.density_fluctuations)

    print(turbulence.Pm)


if __name__ == "__main__":
    tests()