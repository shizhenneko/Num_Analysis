import numpy as np

def f(x):
    return np.sin(np.log(1/(1 + x**2)))

def legendre_poly(n, x):
    if n == 0:
        return np.ones_like(x) if isinstance(x, np.ndarray) else 1
    elif n == 1:
        return x    
    P_prev2 = np.ones_like(x) if isinstance(x, np.ndarray) else 1  # P_0
    P_prev1 = x  # P_1
    for k in range(2, n + 1):
        P_current = ((2*k - 1) * x * P_prev1 - (k - 1) * P_prev2) / k
        P_prev2 = P_prev1
        P_prev1 = P_current
    return P_prev1

def Gauss_Legendre_Quadrature(a,b,n):
    nodes = []
    for i in range(1, n + 1):
        x0 = np.cos(np.pi * (i - 0.25) / (n + 0.5))
        
        for _ in range(100):  
            P_n = legendre_poly(n, x0)
            P_n_minus_1 = legendre_poly(n - 1, x0)
            P_n_derivative = n * (P_n_minus_1 - x0 * P_n) / (1 - x0**2)
            x1 = x0 - P_n / P_n_derivative
            if abs(x1 - x0) < 1e-15:
                break
            x0 = x1
        nodes.append(x1)
    nodes = np.array(nodes)
    # 2. 计算权重
    weights = []
    for x_i in nodes:
        P_n_minus_1 = legendre_poly(n - 1, x_i)
        # 权重公式: w_i = 2 / ((1 - x_i^2) * [P'_n(x_i)]^2)
        # 其中 P'_n(x_i) = n * (P_{n-1}(x_i) - x_i * P_n(x_i)) / (1 - x_i^2)
        # 但由于 P_n(x_i) = 0 (x_i 是根), 所以 P'_n(x_i) = n * P_{n-1}(x_i) / (1 - x_i^2)
        w_i = 2 / ((1 - x_i**2) * (legendre_poly(n - 1, x_i))**2)
        weights.append(w_i)
    
    weights = np.array(weights)
    
    # 3. 区间变换：将 [a, b] 映射到 [-1, 1]
    # x = (b-a)/2 * t + (b+a)/2, 其中 t ∈ [-1, 1]
    # dx = (b-a)/2 * dt
    transformed_nodes = (b - a) / 2 * nodes + (b + a) / 2
    
    # 4. 计算积分
    result = (b - a) / 2 * np.sum(weights * f(transformed_nodes))
    
    return result

if __name__ == "__main__":
    # 测试代码
    print("使用 Gauss-Legendre 积分计算 ∫_0^1 sin(log(1/(1+x^2))) dx")
    print("=" * 60)
    
    a, b = 0, 1
    
    # 测试不同的节点数
    for n in [2, 3, 4, 5, 6, 8, 10]:
        result = Gauss_Legendre_Quadrature(a, b, n)
        print(f"n = {n:2d}: 积分值 = {result:.15f}")
    
    print("=" * 60)
    
    # 也可以测试其他简单函数来验证正确性
    print("\n验证：计算 ∫_0^1 x^2 dx (精确值 = 1/3 = 0.333...)")
    
    def f_test(x):
        return x**2
    
    # 临时修改被积函数
    import types
    original_f = f
    
    for n in [2, 3, 4]:
        # 手动计算
        nodes = []
        for i in range(1, n + 1):
            x0 = np.cos(np.pi * (i - 0.25) / (n + 0.5))
            for _ in range(100):
                P_n = legendre_poly(n, x0)
                P_n_minus_1 = legendre_poly(n - 1, x0)
                P_n_derivative = n * (P_n_minus_1 - x0 * P_n) / (1 - x0**2)
                x1 = x0 - P_n / P_n_derivative
                if abs(x1 - x0) < 1e-15:
                    break
                x0 = x1
            nodes.append(x1)
        
        nodes = np.array(nodes)
        weights = np.array([2 / ((1 - x_i**2) * (legendre_poly(n - 1, x_i))**2) for x_i in nodes])
        transformed_nodes = (1 - 0) / 2 * nodes + (1 + 0) / 2
        result = (1 - 0) / 2 * np.sum(weights * f_test(transformed_nodes))
        
        error = abs(result - 1/3)
        print(f"n = {n}: 积分值 = {result:.15f}, 误差 = {error:.2e}")