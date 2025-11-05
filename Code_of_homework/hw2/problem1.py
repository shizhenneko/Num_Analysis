import numpy as np
# Gauss消元
def gauss(matrix):
    matrix = matrix.copy()
    m,n = np.shape(matrix)
    for k in range(m-1):
        if abs(matrix[k,k])<1e-10:
            found = False
            for i in range(k+1,m):
                if abs(matrix[i,k])>1e-10:
                    # 使用NumPy的高级索引正确交换行
                    matrix[[k,i]] = matrix[[i,k]]
                    found = True
                    break
            if not found:
                print("Singular matrix")
                return None

        for i in range(k+1,m):
            factor = matrix[i,k]/matrix[k,k]
            matrix[i] = matrix[i] - factor*matrix[k]
        
    return matrix
           

# 回代过程
def solution(final_matrix):
    m,n = np.shape(final_matrix)
    x = np.zeros(n-1)
    for i in range(m-1, -1, -1):
        x[i] = final_matrix[i,n-1]
        for j in range(i+1, n-1):
            x[i] = x[i] - final_matrix[i,j]*x[j]
        x[i] = x[i] / final_matrix[i,i]
    return x


if __name__=="__main__":
    matrix = np.array([[2,2,3,3],[4,7,7,1],[-2,4,5,-7]],dtype=np.float64)
    final_matrix = gauss(matrix)
    sol = solution(final_matrix)
    print(sol)