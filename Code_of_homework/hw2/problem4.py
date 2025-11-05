# 此模块需要实现一个Jacobi迭代法，以及一个判断迭代是否收敛的方法
# 还需要实现一个Gauss-Seidal迭代法，以及判断迭代是否收敛的方法
import numpy as np
from typing import Callable

# 计算谱半径(Jacobi迭代)
def jacobi_judge_direct(matrix:np.array):
    matrix = np.copy(matrix)
    n = np.shape(matrix)[0]
    E = np.eye(n)
    D = np.diag(np.diag(matrix))
    Di = np.linalg.inv(D)
    Bj = E - Di@matrix
    eigenvalues = np.linalg.eigvals(Bj)
    abs_eigenvalues = np.abs(eigenvalues)
    spectral_radius = max(abs_eigenvalues)
    if spectral_radius < 1:
        return True
    else:
        return False

#计算谱半径(Gauss-Seidal迭代)
def gauss_seidal_judge_direct(matrix:np.array):
    matrix = np.copy(matrix)
    D = np.diag(np.diag(matrix))
    L = np.tril(matrix,-1)
    U = np.triu(matrix,1)
    inverse = np.linalg.inv(D+L)
    Bg = -inverse@U
    eigenvalues = np.linalg.eigvals(Bg)
    abs_eigenvalues = np.abs(eigenvalues)
    spectral_radius = max(abs_eigenvalues)
    if spectral_radius < 1:
        return True
    else:
        return False

#利用无穷范数和1范数判断Jacobi迭代
def jacobi_judge(matrix:np.array):
    matrix = np.copy(matrix)
    n = np.shape(matrix)[0]
    E = np.eye(n)
    D = np.diag(np.diag(matrix))
    Di = np.linalg.inv(D)
    Bj = E - Di@matrix
    if np.linalg.norm(Bj,np.inf) < 1 or np.linalg.norm(Bj,1) < 1:
        return True
    else:
        return False

#利用无穷范数和1范数判断Gauss-Seidal迭代
def gauss_seidal_judge(matrix:np.array):
    matrix = np.copy(matrix)
    D = np.diag(np.diag(matrix))
    L = np.tril(matrix,-1)
    U = np.triu(matrix,1)
    inverse = np.linalg.inv(D+L)
    Bg = -inverse@U
    if np.linalg.norm(Bg,np.inf) < 1 or np.linalg.norm(Bg,1) < 1:
        return True
    else:
        return False

def jacobi_iteration(matrix,b,x0,tol=1e-8,max_iter = 10000):
    matrix = np.copy(matrix)
    x0 = np.copy(x0)
    D= np.diag(np.diag(matrix))
    Di = np.linalg.inv(D)
    L = np.tril(matrix,-1)
    U = np.triu(matrix,1)
    norm = 1.0
    x = x0
    for iteration in range(max_iter):
       x_new = -Di@(L+U)@x + Di@b 
       norm = np.linalg.norm(x_new - x,np.inf)
       if norm < tol:
           break
       x = x_new
    return x,iteration+1


    



def gauss_seidal_iteration(matrix,b,x0,tol=1e-8,max_iter = 10000):
    matrix = np.copy(matrix)
    x0 = np.copy(x0)
    Bg = -np.linalg.inv(np.tril(matrix)) @ np.triu(matrix,1)
    DLib = np.linalg.inv(np.tril(matrix)) @ b
    norm = 1.0
    x = x0
    for iteration in range(max_iter):
        x_new = Bg@x + DLib
        norm = np.linalg.norm(x_new - x,np.inf)
        if norm < tol:
            break
        x = x_new
    return x,iteration+1

def start(A:np.array,b:np.array,x0:np.array,method:Callable,tol=1e-8,max_iter = 10000):
    if method == jacobi_iteration:
        return jacobi_iteration(A,b,x0,tol,max_iter)
    elif method == gauss_seidal_iteration:
        return gauss_seidal_iteration(A,b,x0,tol,max_iter)
    else:
        raise ValueError("Invalid method")

if __name__ == "__main__":
    A = np.array([[2,-1,1],[1,1,1],[1,1,-2]],dtype = np.float64)
    b = np.array([1,1,1])
    x0 = np.array([0,0,0])
    valid = False
    
    if jacobi_judge_direct(A):
        print("Jacobi iteration is valid")
        solution1,iteration = start(A,b,x0,jacobi_iteration)
        valid = True
        print(solution1)
        print(iteration)
    if gauss_seidal_judge_direct(A):
        print("Gauss-Seidal iteration is valid")
        solution2,iteration = start(A,b,x0,gauss_seidal_iteration)
        valid = True
        print(solution2)
        print(iteration)
    if valid == False:
        print("Can't be iterated")
