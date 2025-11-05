import numpy as np

def f(x):
    return np.exp(-x)/2

def calculate(x0,max_iter=100,epsilon = 1e-6):
    for i in range(max_iter):
        x1 = f(x0)
        print(x1)
        if abs(x1-x0)<epsilon:
            break
        x0=x1
    return x1,i

if __name__ == "__main__":
    x0=0.5
    answer,iter = calculate(x0)
    print("Final answer is {0}, iterations is {1}".format(answer,iter))