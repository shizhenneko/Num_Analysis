from typing import Callable
import numpy as np
from scipy import integrate
import math

def g(x):
    return x

def f(x):
    return np.sin(x)

#实现一个Newton-Cotes插值计算积分的算法
#输入：积分区间[a,b]，积分节点最大序号数n
#输出:一组机械求积公式的权重
def Newton_Cotes_weight(a,b,n):
    weight = np.zeros(n+1)
    y = np.zeros(n+1)
    for j in range(n+1):
        # C_j = H_j / (b-a) = (-1)^n-j /(n * j! * (n-j)!) ∫0 n (t-i)dt  i=0,1,...,n-1,n and i≠j
        C_j = (-1)**(n-j) / (n * math.factorial(j) * math.factorial(n-j)) * integrate.quad(lambda t: basis_product(t,j,n), 0, n)[0]
        weight[j] = C_j 
    return weight

#实现辅助函数 basis_product(x,j) = (x-0)*(x-1)*...*(x-n)/(x-j)
def basis_product(x_val,j,n):
    result = 1
    for i in range(n+1):
        if i != j:
            result *= (x_val - i)
    return result

# 将函数值转为对应序列
def Newton_Cotes_function(a,b,n,f):
    y = np.zeros(n+1)
    h = (b-a)/n
    for i in range(n+1):
        y[i] = f(a + h*i)

    return y


#实现一个Newton-Cotes插值计算积分的算法
#输入：积分区间[a,b]，积分节点最大序号数n,函数节点值,节点值要比n大1
#输出:积分值
def Newton_Cotes_integral(a,b,n,f:Callable):
    weight = Newton_Cotes_weight(a,b,n)
    y = Newton_Cotes_function(a,b,n,f)
    integral = (b-a)*sum(weight[i]*y[i] for i in range(n+1))
    return integral

if __name__ == "__main__":
    result = Newton_Cotes_integral(0,2,1,g)
    print(result)
