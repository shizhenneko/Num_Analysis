import numpy as np
from typing import Callable
def Doolittle(matrix: np.array) -> (np.array, np.array):
    matrix = np.copy(matrix)
    n = np.shape(matrix)[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        L[i, i] = 1  # L的对角线为1
        
        # 计算U的第i行（从第i列到第n-1列）
        for j in range(i, n):
            U[i, j] = matrix[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        
        # 计算L的第i列（从第i+1行到第n-1行）
        for j in range(i + 1, n):
            L[j, i] = (matrix[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    print(L) 
    print(U)
    return L, U


def Crout(matrix:np.array)->np.array:
    matrix = np.copy(matrix)
    n = np.shape(matrix)[0]
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    for i in range(n):
        U[i,i] = 1
        for j in range(i,n):
            L[j,i] = matrix[j,i] - sum(L[j,k]*U[k,i] for k in range(i))
        for j in range(i+1,n):
            U[i,j] = (matrix[i,j] - sum(L[i,k]*U[k,j] for k in range(i))) / L[i,i]
    print(L)
    print(U)
    return L,U

def Doolittle_solve(L:np.array,U:np.array,b:np.array)->np.array:
    n = np.shape(L)[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i,k]*y[k] for k in range(i))
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = (y[i] - sum(U[i,k]*x[k] for k in range(i+1,n))) / U[i,i]
    return x

    

def Crout_solve(L:np.array,U:np.array,b:np.array)->np.array:
    n = np.shape(L)[0]
    z = np.zeros(n)
    for i in range(n):
        z[i] = (b[i] - sum(L[i,k]*z[k] for k in range(i))) / L[i,i]
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = (z[i] - sum(U[i,k]*x[k] for k in range(i+1,n)))
    return x


def start(matrix:np.array,b:np.array,method:Callable)->np.array:
    if method == Doolittle:
        L,U = Doolittle(matrix)
        return Doolittle_solve(L,U,b)
    elif method == Crout:
        L,U = Crout(matrix)
        return Crout_solve(L,U,b)
    else:
        raise ValueError("Invalid method")

if __name__=="__main__":
    matrix = np.array([[4,-2,4],[-2,17,10],[-4,10,9]],dtype = np.float64)
    b = np.array([10,3,-7],dtype = np.float64)
    solution1=start(matrix,b,Doolittle)
    solution2=start(matrix,b,Crout)
    print(solution1)
    print(solution2)