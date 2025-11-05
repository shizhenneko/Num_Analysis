import numpy as np

def gauss_jordan(matrix: np.array) -> np.array:
    n = np.shape(matrix)[0]
    for i in range(n):
        # 选主元：如果当前主元太小，交换行
        if abs(matrix[i,i])<1e-10:
            for j in range(i+1,n):
                if abs(matrix[j,i])>1e-10:
                    matrix[[i,j]] = matrix[[j,i]]
                    break
        
        # 关键步骤：将主元归一化为1
        pivot = matrix[i,i]
        matrix[i] = matrix[i] / pivot
        
        # 消元：将其他行的第i列元素变为0
        for j in range(n):
            if j!=i:
                factor = matrix[j,i]
                matrix[j] = matrix[j] - factor*matrix[i]

    return matrix

# 原始矩阵
A = np.array([[1,1,-1],[2,1,0],[1,-1,0]], dtype=np.float64)
print("原始矩阵 A:")
print(A)
print()

# 构造增广矩阵 [A | I]
n = A.shape[0]
augmented_matrix = np.hstack([A, np.eye(n)])

# 应用高斯-约当消元
final_matrix = gauss_jordan(augmented_matrix)

# 提取逆矩阵（增广矩阵的右半部分）
inverse_matrix = final_matrix[:, n:]

print("逆矩阵 A^(-1):")
print(inverse_matrix)
print()

# 验证：A * A^(-1) 应该等于 I
verification = A @ inverse_matrix
print("验证 A * A^(-1):")
print(verification)
print()
print("是否接近单位矩阵:", np.allclose(verification, np.eye(n)))