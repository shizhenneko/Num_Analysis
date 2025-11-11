# lab4-1
# 使用n为[5, 10, 15, 20, 25, 30]
import numpy as np
from typing import Callable


def f1(x):
    return np.sin(x)
def f2(x):
    return (x**2)
def gauss_laguerre(f:Callable,n:int):
    
    nodes, weights = np.polynomial.laguerre.laggauss(n)
    result = np.sum(weights * f(nodes))
    return result

def gauss_hermite(f:Callable,n:int):   
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    result = np.sum(weights * f(nodes))
    return result


def main():
    for n in [5, 10, 15, 20, 25, 30]:
        print("for n = {0}".format(n))
        result1 = gauss_laguerre(f1,n)
        result2 = gauss_hermite(f2,n)
        print("problem 1 use Gauss Laguerre: result is {0:.5f}".format(result1))
        print("problem 2 use Gauss Hermite: result is {0:.5f}".format(result2))
    


if __name__ == "__main__":
    main()

