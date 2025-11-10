#完成一个复化的梯形公式算法
 
from typing import Callable
import numpy as np
from Newton_Cotes import Newton_Cotes_integral


def f(x):
    return np.sin(x)

#完成对于复化过程的刻画
#给出区间[a,b]以及等分数n，给出复化区间
def complexified(a,b,n):
    complexified_intervals = np.zeros((n,2))
    h = (b-a)/n
    for i in range(n):
        x_i = a+i*h
        x_j = a+(i+1)*h
        complexified_intervals[i,0] = x_i
        complexified_intervals[i,1] = x_j
    return complexified_intervals

#给出子区间，想要使用Newton-Cotes的阶数以及函数
def Complex_Newton_Cotes(interval:np.array,n,f:Callable):
    result = 0
    for i in range(len(interval)):
        result += Newton_Cotes_integral(interval[i][0],interval[i][1],n,f)
    return result

if __name__ == "__main__":
    interval = complexified(0,2,1)
    result = Complex_Newton_Cotes(interval,1,f)
    print(result)
        

