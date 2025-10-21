import numpy as np

def f0(x):
    return np.sin(x)+(x-np.pi)

def f1(x):
    return np.cos(x)+1

x=3.5
k=1

y=x-k*f0(x)/f1(x)
z=y-k*f0(y)/f1(y)

print("Initial step =",x)
print("First step =", y)
print("Second step =",z)

theta= (x*z - y**2) / (z-2*y+x)
r = (theta - y)/(z-y)

print("r=",r)





