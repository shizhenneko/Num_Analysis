#完成一个最佳逼近的算法
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 先定义一个想要逼近的多项式
def f(x):
    return np.sqrt(x)

def g(x):
    return np.sin(x)

# 默认使用{1,x,x^2....}给出想要逼近的最高次数，以及内积区间
def optimal_approximation(function,n,a,b):
    # 先计算内积矩阵 A_{(n+1)\times(n+1)}
    A = np.zeros((n+1,n+1))
    # 向量 v_{(n+1)\times 1}
    v = np.zeros(n+1)
    # 向量 x_{(n+1)\times 1} 解向量
    x = np.zeros(n+1)
    for i in range(n+1):
        for j in range(n+1):
            # A[i,j] = <\varphi_i,\varphi_j> = ∫[a,b] x^i * x^j dx
            # 定义基函数 phi_i(x) = x^i, phi_j(x) = x^j
            phi_i = lambda x, power=i: x**power
            phi_j = lambda x, power=j: x**power
            A[i,j] = inner_product(phi_i, phi_j, a, b)

    for i in range(n+1):
        # v[i] = <f,\varphi_i> = ∫[a,b] f(x) * x^i dx
        phi_i = lambda x, power=i: x**power
        v[i] = inner_product(function, phi_i, a, b)
    
    # 解线性方程组 Ax = v，得到系数组x
    x = np.linalg.solve(A, v)
    return x

# 此处我们要定义函数的内积运算，即在[a,b]区间内的积分
def inner_product(f, g, a, b):
    integrand = lambda x: f(x) * g(x)
    result,_ = integrate.quad(integrand, a, b)
    return result


# 根据系数数组重建多项式
def reconstruct_polynomial(coeffs):
    def poly(x):
        result = 0
        for i, coeff in enumerate(coeffs):
            result += coeff * (x ** i)
        return result
    return poly

# 打印多项式表达式
def print_polynomial(coeffs):
    terms = []
    for i, coeff in enumerate(coeffs):
        if abs(coeff) < 1e-10:  # 忽略接近0的系数
            continue
        if i == 0:
            terms.append(f"{coeff:.6f}")
        elif i == 1:
            if coeff >= 0:
                terms.append(f"+ {coeff:.6f}x")
            else:
                terms.append(f"- {abs(coeff):.6f}x")
        else:
            if coeff >= 0:
                terms.append(f"+ {coeff:.6f}x^{i}")
            else:
                terms.append(f"- {abs(coeff):.6f}x^{i}")
    
    if not terms:
        return "0"
    
    poly_str = terms[0]
    for term in terms[1:]:
        poly_str += " " + term
    
    return poly_str


# 测试示例
if __name__ == "__main__":
   n = 1
   a = 0
   b = 1
   x = optimal_approximation(f,n,a,b)
   print("系数数组:", x)
   print("\n逼近多项式:")
   print("P(x) =", print_polynomial(x))
   
   # 重建多项式函数
   P = reconstruct_polynomial(x)
   
   # 绘制原函数和逼近多项式的对比图
   x_plot = np.linspace(a, b, 1000)
   y_original = f(x_plot)
   y_approx = P(x_plot)
   
   plt.figure(figsize=(10, 6))
   plt.plot(x_plot, y_original, 'b-', label='原函数 f(x)', linewidth=2)
   plt.plot(x_plot, y_approx, 'r--', label=f'逼近多项式 P_{n}(x)', linewidth=2)
   plt.xlabel('x')
   plt.ylabel('y')
   plt.title(f'最佳平方逼近 (n={n}, 区间=[{a},{b}])')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()
   
   # 计算逼近误差
   error = np.sqrt(inner_product(lambda t: (f(t) - P(t))**2, lambda t: 1, a, b))
   print(f"\n均方误差: {error:.10f}")