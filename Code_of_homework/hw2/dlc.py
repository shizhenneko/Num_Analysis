#首先我们需要完成一个最速下降算法
#接着完成一个共轭梯度算法

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

#定义出一个多项式，根据给定的矩阵和向量，计算出函数值
# f(x) = 0.5*x^T*A*x - b^T*x,其中x为列向量，@自动决定行还是列向量
def f(matrix:np.array,b:np.array,x:np.array):
    f = 0.5*(x @ matrix @ x - 2*b @ x)
    return f
# 完成一个最速下降算法
def bfgs_method(matrix:np.array,b:np.array,x0:np.array,tol=1e-6,max_iter=100):
    matrix= np.copy(matrix)
    x = np.copy(x0)
    data = np.array([x0])
    for i in range(max_iter):
        # s = b - Ax = -\ grad f(x)
        s = b - matrix @ x
        # alpha = (s^T @ s) / (s^T @ A @ s)
        alpha = (s @ s) / (s @ matrix @ s)
        # x_new = x + alpha * s
        x_new = x + alpha * s
        data = np.vstack((data, x_new))
        if np.linalg.norm(x_new - x,np.inf) < tol:
            x = x_new
            break
        x = x_new
    print('Total interation step = ',i+1)
    print("Numerical minimum function value = ",f(matrix,b,x))
    return x,data

# 可视化这个最速下降算法
def visualize_bfgs_method_2_dimension(matrix:np.array, b:np.array, x0:np.array, x_range=(-5, 5), y_range=(-5, 5)):
    """
    可视化最速下降算法的迭代过程
    
    参数:
    matrix: 二次型矩阵 A
    b: 线性项系数向量
    x0: 初始点
    x_range: x轴范围
    y_range: y轴范围
    """
    x, data = bfgs_method(matrix, b, x0)
    # 划定范围
    xx = np.arange(x_range[0], x_range[1], 0.05)
    yy = np.arange(y_range[0], y_range[1], 0.05)
    # 创建网格用于绘制等高线
    p, q = np.meshgrid(xx, yy)
    #计算函数值 - 对网格上每个点计算函数值
    r = np.zeros_like(p)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            r[i, j] = f(matrix, b, np.array([p[i, j], q[i, j]]))
    
    print('等高线图最小值 =', np.min(r))
    
    # 创建图形(10x8英寸窗口)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制等高线(p,q分别为上面的x,y坐标，r为函数值，20条等高线)
    CS = ax.contour(p, q, r, levels=20)
    #标注数值(inline=True表示在等高线上标注数值),fontsize为字体大小
    ax.clabel(CS, inline=True, fontsize=10)
    #设置标题(最速下降法迭代过程),fontsize为字体大小，fontweight为字体粗细
    ax.set_title('最速下降法迭代过程', fontsize=14, fontweight='bold')
    
    # 绘制最终解(x为最终解，marker为标记，s为大小，c为颜色，edgecolors为边框颜色，linewidths为线宽，zorder为层级)
    ax.scatter(x[0], x[1], marker='*', s=200, c='red', 
               edgecolors='black', linewidths=2, zorder=30,
               label=f'最终解 ({x[0]:.4f}, {x[1]:.4f})')
    
    # 绘制初始点
    ax.scatter(x0[0], x0[1], marker='o', s=100, c='green', 
               edgecolors='black', linewidths=2, zorder=25,
               label=f'初始点 ({x0[0]:.2f}, {x0[1]:.2f})')
    
    # 绘制迭代路径
    ax.plot(data[:, 0], data[:, 1], '^-', c='blue', linewidth=2, 
            markersize=6, zorder=20, label=f'迭代路径 (共{len(data)}步)')
    
    # 添加箭头显示方向
    for i in range(len(data)-1):
        ax.annotate('', xy=data[i+1], xytext=data[i],
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.6))
    
    # 设置图例和网格
    ax.legend(loc='best', fontsize=10)
    ax.grid(c='lightgray', linestyle='--', alpha=0.7)
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    
    # 设置坐标轴范围
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    fig.tight_layout()
    
    # 可选：保存图像
    # fig.savefig("steepest_descent.pdf")
    # fig.savefig("steepest_descent.png", dpi=300)
    
    plt.show()
    return x,data

def conjugate_gradient(matrix:np.array,b:np.array,x0:np.array,tol=1e-6,max_iter=100):
    matrix= np.copy(matrix)
    x = np.copy(x0)
    data = np.array([x0])
    #最速下降一次
    r0 = b - matrix @ x
    alpha0 = (r0 @ r0) / (r0 @ matrix @ r0)
    x = x + alpha0 * r0
    data = np.vstack((data, x))
    # 共轭梯度一次
    r1 = b- matrix @ x
    beta0 = -(r1 @ matrix @ r0)/(r0 @ matrix @r0)
    p1 = r1 + beta0 * r0

    alpha1 = (r1 @ p1)/(p1 @ matrix @ p1)
    x = x+ alpha1 * p1
    data = np.vstack((data, x))
    return x,data

def visualize_conjugate_gradient(matrix:np.array,b:np.array,x0:np.array,x_range=(-5, 5),y_range=(-5, 5)):
    x,data = conjugate_gradient(matrix,b,x0)
    xx = np.arange(x_range[0],x_range[1],0.05)
    yy = np.arange(y_range[0],y_range[1],0.05)
    p,q = np.meshgrid(xx,yy)
    r = np.zeros_like(p)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            r[i,j] = f(matrix,b,np.array([p[i,j],q[i,j]]))
    print('等高线图最小值 =',np.min(r))
    fig,ax = plt.subplots(figsize=(10,8))
    CS = ax.contour(p,q,r,levels=20)
    ax.clabel(CS,inline=True,fontsize=10)
    ax.set_title('共轭梯度法迭代过程',fontsize=14,fontweight='bold')
    ax.scatter(x[0],x[1],marker='*',s=200,c='red',edgecolors='black',linewidths=2,zorder=30,label=f'最终解 ({x[0]:.4f}, {x[1]:.4f})')
    ax.scatter(x0[0],x0[1],marker='o',s=100,c='green',edgecolors='black',linewidths=2,zorder=25,label=f'初始点 ({x0[0]:.2f}, {x0[1]:.2f})')
    ax.plot(data[:,0],data[:,1],'^-',c='blue',linewidth=2,markersize=6,zorder=20,label=f'迭代路径 (共{len(data)}步)')
    for i in range(len(data)-1):
        ax.annotate('',xy=data[i+1],xytext=data[i],arrowprops=dict(arrowstyle='->',color='blue',lw=1.5,alpha=0.6))
    ax.legend(loc='best',fontsize=10)
    ax.grid(c='lightgray',linestyle='--',alpha=0.7)
    ax.set_xlabel(r'$x_1$',fontsize=12)
    ax.set_ylabel(r'$x_2$',fontsize=12)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    fig.tight_layout()

    plt.show()
    return x,data


def start(A:np.array,b:np.array,x0:np.array,method:Callable):
    if method == "bfgs":
        return visualize_bfgs_method_2_dimension(A,b,x0)
    elif method == "conjugate_gradient":
        return visualize_conjugate_gradient(A,b,x0)
    else:
        raise ValueError("Invalid method")

if __name__== "__main__":

    A = np.array([[0.5, 0], [0, 2.5]])
    b = np.array([0, 0])
    x = np.array([3.7, 2.5])
    start(A,b,x,"bfgs")
    start(A,b,x,"conjugate_gradient")

