import numpy as np

def create_tridiagonal_matrix(n):
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = 2
        if i > 0:
            A[i,i-1] = -1
        if i < n-1:
            A[i,i+1] = -1
    return A

def create_rhs_vector(n):
    return np.full(n,4.0/(n+1)**2)

def jacobi_iteration(A,b,x0,tol=1e-8,max_iter = 10000):
    n = len(b)    
    x = x0.copy()
    x_new = np.zeros(n)
    iterations = 0
    for iterations in range(max_iter):
        for i in range(n):
            sum_ax = 0
            for j in range(n):
                if j!= i:
                    sum_ax += A[i,j]*x[j]
            x_new[i] = (b[i] - sum_ax) / A[i,i] 

        if np.linalg.norm(x_new - x, np.inf) < tol:
            break

        x = x_new.copy()

    return x, iterations + 1

def gauss_seidal_iteration(A,b,x0,tol=1e-8,max_iter = 10000):
    n = len(b)
    x = x0.copy()
    iterations = 0

    for iterations in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            sum_ax = 0
            for j in range(i):
                sum_ax += A[i,j] * x[j]

            for j in range(i+1,n):
                sum_ax += A[i,j] * x_old[j]
            x[i] = (b[i] - sum_ax) / A[i,i]

        if np.linalg.norm(x-x_old,np.inf)<tol:
            break
    
    return x,iterations+1

def solve_for_different_n():
    n_values = [5,11,21]

    print("=" * 80)
    print("线性方程组求解结果")
    print("=" * 80)

    for n in n_values:
        print(f"\n当 n = {n} 时:")
        print("-" * 50)

        A = create_tridiagonal_matrix(n)
        b = create_rhs_vector(n)
        x0 = np.zeros(n)

        x_jacobi,iter_jacobi = jacobi_iteration(A,b,x0)
        norm_jacobi = np.linalg.norm(x_jacobi,np.inf)

        x_gs, iter_gs = gauss_seidal_iteration(A,b,x0)
        norm_gs = np.linalg.norm(x_gs,np.inf)

        print(f"Jacobi迭代法:")
        print(f"迭代次数: {iter_jacobi}")
        print(f"解的无穷范数: {norm_jacobi:.10f}")
        print(f"解向量: {x_jacobi}")

        print(f"\nGauss-Seidal迭代法:")
        print(f"迭代次数:{iter_gs}")
        print(f"解的无穷范数: {norm_gs:.10f}")
        print(f"解向量: {x_gs}")


if __name__ == "__main__":
    solve_for_different_n()
    



