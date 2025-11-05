# 1. 利用内建函数生成[-1,1]区间上n次Chebyshev多项式的零点
# 2. 推导[-5,5]关于1/\sqrt(1-x^2) 的11次正交多项式的零点

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 100  # 设置图像清晰度

# 任务1: 生成[-1,1]区间上n次Chebyshev多项式的零点
def chebyshev_zeros_standard(n):
    # 方法1: 使用解析公式
    # n次Chebyshev多项式的零点: x_k = cos((2k-1)π/(2n)), k=1,2,...,n
    k = np.arange(1, n + 1)
    # 向量化操作
    zeros_formula = np.cos((2 * k - 1) * np.pi / (2 * n))
    return np.sort(zeros_formula)
# 任务2: 推导[-5,5]关于1/sqrt(1-x^2)的11次正交多项式的零点
def chebyshev_zeros_transformed(n, a, b):
    # 先求[-1,1]上n次Chebyshev多项式的零点
    zeros_standard = chebyshev_zeros_standard(n)
    # 线性变换到[a,b]区间
    # t ∈ [-1,1] -> x ∈ [a,b]
    # x = (b-a)/2 * t + (a+b)/2
    zeros_transformed = (b - a) / 2 * zeros_standard + (a + b) / 2
    return np.sort(zeros_transformed)

# 定义插值函数
def f(x):
    return 1/(1+x**2)

# 给出插值条件以及插值次数,先计算基函数
def lagrange_interpolation(x: np.array, y: np.array):
    n = np.shape(x)[0]
    # 基函数
    def lagrange_base_function(i, x_val):
        result = 1.0
        for j in range(n):
            if j != i:
                result *= (x_val - x[j]) / (x[i] - x[j])
        return result
    # 插值函数
    def L(x_val):
        result = 0.0
        for i in range(n):
            result += y[i] * lagrange_base_function(i, x_val)
        return result
    # 返回一个函数
    return L

def lagrange_visualization(L, x, y, a=-5, b=5):
    """
    可视化Lagrange插值结果
    
    参数:
        L: 插值函数
        x: 插值节点
        y: 节点处的函数值
        a, b: 绘图区间
    """
    # 生成密集的绘图点
    x_plot = np.linspace(a, b, 500)
    y_true = np.array([f(xi) for xi in x_plot])  # 真实函数值
    y_interp = np.array([L(xi) for xi in x_plot])  # 插值结果
    
    # 创建图形，调整大小为更适中的尺寸
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制真实函数曲线 - 使用深蓝色，较粗的线条
    ax.plot(x_plot, y_true, color='#2E86AB', linewidth=2.5, 
            label=r'真实函数 $f(x) = \frac{1}{1+x^2}$', zorder=2)
    
    # 绘制插值多项式曲线 - 使用橙红色，虚线
    ax.plot(x_plot, y_interp, color='#E63946', linestyle='--', 
            linewidth=2.5, label=f'Lagrange插值多项式 (n={len(x)-1})', zorder=3)
    
    # 绘制插值节点 - 使用绿色圆点，带白色边框
    ax.plot(x, y, 'o', color='#06A77D', markersize=9, 
            markeredgecolor='white', markeredgewidth=1.5, 
            label=f'Chebyshev零点 ({len(x)}个节点)', zorder=4)
    
    # 设置坐标轴标签
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')
    
    # 设置标题
    ax.set_title('基于Chebyshev零点的Lagrange插值', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # 添加图例，位置自动选择最佳位置
    ax.legend(fontsize=12, loc='best', framealpha=0.95, 
              edgecolor='gray', fancybox=True, shadow=True)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 设置坐标轴范围
    ax.set_xlim([a, b])
    y_min, y_max = min(y_true.min(), y_interp.min()), max(y_true.max(), y_interp.max())
    y_range = y_max - y_min
    ax.set_ylim([y_min - 0.05*y_range, y_max + 0.05*y_range])
    
    # 美化刻度
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 添加背景色
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # 添加误差分析文本
    max_error = np.max(np.abs(y_true - y_interp))
    mean_error = np.mean(np.abs(y_true - y_interp))
    
    # 在图上添加误差信息
    textstr = f'最大误差: {max_error:.2e}\n平均误差: {mean_error:.2e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细信息
    print("=" * 60)
    print("插值结果分析")
    print("=" * 60)
    print(f"插值节点数量: {len(x)}")
    print(f"插值多项式次数: {len(x)-1}")
    print(f"插值区间: [{a}, {b}]")
    print(f"最大误差: {max_error:.6e}")
    print(f"平均误差: {mean_error:.6e}")
    print(f"相对误差: {max_error/np.max(np.abs(y_true)):.6e}")
    print("=" * 60)



# 测试和显示结果
if __name__ == "__main__":
    # 任务1: 显示[-1,1]区间上的Chebyshev零点
    print("任务1: [-1,1]区间上的Chebyshev多项式零点")
    zeros_standard = chebyshev_zeros_standard(11)
    print(f"11次Chebyshev多项式的零点:\n{zeros_standard}\n")
    
    # 任务2: 显示[-5,5]区间上的Chebyshev零点
    print("任务2: [-5,5]区间上的Chebyshev多项式零点")
    zeros = chebyshev_zeros_transformed(11, -5, 5)
    print(f"变换后的零点:\n{zeros}\n")
    
    # 使用Chebyshev零点进行Lagrange插值
    y = np.array([f(xi) for xi in zeros])
    L = lagrange_interpolation(zeros, y)
    
    # 可视化插值结果
    lagrange_visualization(L, zeros, y, a=-5, b=5)
