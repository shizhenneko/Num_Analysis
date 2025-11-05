import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.5, 0.2], [0.8, 3]])
b = np.array([-0.3, 0.4])
c = -1
# A = np.array([[0.5, 0], [0, 2.5]])
# b = np.array([0, 0])
# c = 0

x = np.array([3.7, 2.5])
data = np.array([x])

for i in range(100):
    # s = - (2 * A @ x + b)
    s = - ((A + A.T) @ x + b)
    k1 = s @ A @ s
    k2 = x @ A @ s + s @ A @ x + b @ s
    # k2 = s @ s
    alpha = - 0.5 * k2 / k1
    y = x + alpha * s
    data = np.row_stack((data, y))
    if np.linalg.norm(y - x, np.inf) < 10 ** -6:
        x = y
        print('Final solution = ', x)
        break
    x = y
print('Total iteration step =', len(data))
print('Numerical minimum function value =', x @ A @ x + b @ x + c)

xx = np.arange(-4.1, 4.1, 0.01)
yy = np.arange(-4.1, 4.1, 0.01)
p, q = np.meshgrid(xx, yy)
r = 0.5 * p ** 2 + 3 * q ** 2 + 1 * p * q - 0.3 * p + 0.4 * q - 1
# r = 0.5 * p ** 2 + 2.5 * q ** 2
print('Min =', np.min(r))

fig, ax = plt.subplots()
CS = ax.contour(p, q, r)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Contour figure')
ax.scatter(x[0], x[1], marker='o', s=80, c='b', zorder=20,
           label=r'Final solution')
ax.plot(data[:, 0], data[:, 1], '^-', c='r', zorder=30,
           label=r'Iteration steps')
ax.legend(loc='upper left')  # 显示图例说明
ax.grid(c='lightgray', linestyle='--')  # 设置网格线样式
ax.set_xlabel(r'$x$')  # 设置横坐标的名称
ax.set_ylabel(r'$y$')  # 设置纵坐标的名称

fig.tight_layout()  # 设置紧凑显示图像
# fig.savefig("p1.pdf")
# fig.savefig("p1.eps")
plt.show()  # 显示图像
