import numpy as np

def f(x):
    return (x**3 - 5)**2

def f1(x):
    return 2*(x**3 - 5)*3*x**2

def calculate(x0,max_iter = 1000,epsilon = 1e-6):
    r = 2
    print("Initial step = {0}".format(x0))
    for i in range(max_iter):
        x1 = x0 - r*f(x0)/f1(x0)
        print("Step {0} = {1}".format(i+1,x1))
        if abs(x1-x0)<epsilon:
            break
        x0 = x1
    return x1,i+1

if __name__ == "__main__":
    x0 = 1.8
    max_iter = 4
    value,iteration = calculate(x0,max_iter)
    print("最终结果:{value},循环次数:{iter}".format(value=value,iter=iteration))