# 实现一个Lagrange插值方法
import numpy as np
import matplotlib.pyplot as plt

# 原函数
def f(x):
    return np.log(x)

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

def lagrange_visualization(L,x,y):
    x_plot = np.linspace(0.4, 0.8, 100)
    y_true = np.array([f(xi) for xi in x_plot])  # 真实函数值
    y_interp = np.array([L(xi) for xi in x_plot])  # 插值结果
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true, 'b-', label='真实函数 f(x)=ln(x)', linewidth=2)
    plt.plot(x_plot, y_interp, 'r--', label='Lagrange插值多项式', linewidth=2)
    plt.plot(x, y, 'go', markersize=10, label='插值节点')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Lagrange插值', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 插值节点
    x = np.array([0.4, 0.5, 0.7, 0.8], dtype=np.float64)
    y = np.array([-0.916291, -0.693147, -0.356675, -0.223144], dtype=np.float64)
    
    # 构造Lagrange插值多项式
    L = lagrange_interpolation(x, y)
    print(L(0.6))
    lagrange_visualization(L,x,y)
