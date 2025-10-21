import numpy as np
from numpy import log

def f(x):
    return log(x)-1

def bisection_method(a,b,tol):
    if f(a)*f(b)>=0:
        print("No root in the interval")
        return None
    while (b-a)/2>tol:
        c=(a+b)/2
        if f(c)==0:
            return c
        elif f(a)*f(c)<0:
            b=c
        else:
            a=c
    return (a+b)/2

print(bisection_method(2,3,0.5*1e-6))

