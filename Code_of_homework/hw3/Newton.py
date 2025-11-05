import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 9*x**5 + 4
#构造一个牛顿插值法
def Newton(x:np.array,y:np.array):
    n = np.shape(x)[0]
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            diff_table[i][j] = (diff_table[i][j-1] - diff_table[j-1][j-1]) / (x[i] - x[j-1])
    
    def Newton_interpolation(x_val):
        result = diff_table[n-1][n-1]
        for i in range(n-2, -1, -1):
            result = result * (x_val - x[i]) + diff_table[i][i]
        return result
    
    return Newton_interpolation

if __name__ == "__main__":
    x = np.array([0, 1, 2, 3, 4,5], dtype=float)
    y = np.array([f(xi) for xi in x])
    N=Newton(x,y)