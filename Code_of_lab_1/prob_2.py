import numpy as np
from numpy import log

def f(x):
    return log(x)-1;

def f_prime(x):
    return 1/x;

def newton_method(x0,tol=1e-6,max_iter = 100):
    x=x0
    for i in range(max_iter):
        print('step :', i+1)
        y = x - f(x)/f_prime(x)
        print (y)
        if np.abs(y-x)<tol:
            x=y
            break
        x=y
    return y
print("the solution of root =")
print(newton_method(1.5))


