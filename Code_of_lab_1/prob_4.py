import numpy as np

def f0(x):
    return np.sin(x) + (x-np.pi)

def f1(x):
    return np.cos(x) + 1

def newton_method(x0,r,tol=1e-6,max_iter=100):
    x=x0
    for i in range(max_iter):
        print('step :', i+1)
        f1_value = f1(x)
        if np.abs(f1_value)<1e-12:
            print("The derivative is 0, so the method is not applicable")
            return x
        y=x-r*f0(x)/f1(x)
        print(y)
        if np.abs(y-x)<tol:
            x=y
            break
        x=y
    return y
print("Numerical solution of root =")
print(newton_method(3.5,3.0))