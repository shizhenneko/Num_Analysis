import numpy as np


def f(x):
    return np.sin(x)

#实现分段线性插值
def linear_interpolation(x:np.array,y:np.array):
    n = len(x)
    
    def help_linear(x_val):
        result = 0.0
        # l_0(x): 在 [x_0, x_1] 区间
        if x[0] <= x_val <= x[1]:
            result += (x_val - x[1]) / (x[0] - x[1]) * y[0]
        
        # l_j(x): j = 1, 2, ..., n-2
        for i in range(1, n-1):
            if x[i-1] <= x_val <= x[i]:
                result += (x_val - x[i-1]) / (x[i] - x[i-1]) * y[i]
            elif x[i] <= x_val <= x[i+1]:
                result += (x_val - x[i+1]) / (x[i] - x[i+1]) * y[i]
        
        # l_{n-1}(x): 在 [x_{n-2}, x_{n-1}] 区间
        if x[n-2] <= x_val <= x[n-1]:
            result += (x_val - x[n-2]) / (x[n-1] - x[n-2]) * y[n-1]
        
        return result
    
    return help_linear


if __name__ == "__main__":
    x = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    y = np.array([f(xi) for xi in x])
    linear_interpolation = linear_interpolation(x, y)
    print(linear_interpolation(np.pi/6))
    print(f(np.pi/6))
    print("误差 =", abs(linear_interpolation(np.pi/6) - f(np.pi/6)))
