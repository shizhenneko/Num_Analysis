import numpy as np
from scipy.linalg import lu

a = np.array([[1,2,1],[2,2,3],[-1,3,0]])
# PA=LU，对矩阵做LU分解
p,L,U = lu(a)
print(p)
print(L)
print(U)
# @是矩阵乘法运算符，Python内置的
print(p@L)