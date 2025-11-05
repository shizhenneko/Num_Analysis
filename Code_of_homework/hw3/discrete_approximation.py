# 完成一个离散的最佳逼近，也就是拟合问题
import numpy as np

# 给出x:np.array y:np.array两个自变量和函数值的数组，给出\rho:np.array这个权函数数组,给出想要拟合的次数n，默认span{1,x,x^2....}

# 定义一个流程
def optimal_approximation_discrete(x:np.array, y:np.array, weight:np.array,n):
    A = np.zeros((n+1,n+1))
    v = np.zeros(n+1)
    sol = np.zeros(n+1)
    for i in range(n+1):
        for j in range(n+1):
        # 定义A[i][j] = <\varphi_i,\varphi_j>
            varphi_i = lambda x,power = i:x**power
            varphi_j = lambda x,power = j:x**power
            A[i][j] = inner_production_var(varphi_i,varphi_j,weight,x)
    
    for i in range(n+1):
        #定义v[i] = <f,\varphi_i>
        varphi_i = lambda x,power = i:x**power
        v[i] = inner_production_y(varphi_i,weight,x,y)
    sol = np.linalg.solve(A,v)
    return sol


# 求离散函数内积
def inner_production_var(f,g,weight:np.array,x:np.array):
    return sum(f(x[i])*g(x[i])*weight[i] for i in range(len(x)))

def inner_production_y(f,weight:np.array,x:np.array,y:np.array):
    return sum(f(x[i])*weight[i]*y[i] for i in range(len(x)))


def print_polynomial(coeffs):
    terms = []
    for i, coeff in enumerate(coeffs):
        if abs(coeff) < 1e-10:  # 忽略接近0的系数
            continue
        if i == 0:
            terms.append(f"{coeff:.6f}")
        elif i == 1:
            if coeff >= 0:
                terms.append(f"+ {coeff:.6f}x")
            else:
                terms.append(f"- {abs(coeff):.6f}x")
        else:
            if coeff >= 0:
                terms.append(f"+ {coeff:.6f}x^{i}")
            else:
                terms.append(f"- {abs(coeff):.6f}x^{i}")
    
    if not terms:
        return "0"
    
    poly_str = terms[0]
    for term in terms[1:]:
        poly_str += " " + term
    
    return poly_str



if __name__ == "__main__":
    x = np.array([1,2,3,4],dtype = np.float64)
    y = np.array([1.9,2.7,2.9,3.5],dtype = np.float64)
    weight = np.array([1,1,1,1],dtype = np.float64)
    n = 1
    sol = optimal_approximation_discrete(x,y,weight,n)
    poly = print_polynomial(sol)
    print(poly)
