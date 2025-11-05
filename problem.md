# Chapter1

## problem 1
$f(x) = e^{-x}-2x$
$f'(x) = -e^{-x}-2<0$单减，$f(0)=1,f(1)<0$一个单根

$x_{i+1} = e^{-x_{i}}/2$ 构造单点迭代在$[0,1]$
由于$\varphi(x) = e^{-x}/2$在$[0,1]$中$\varphi(x)\subset [0,1]$

不妨取初值$x_0 = 0.5$利用Python解得
**Final answer is 0.35173353432626475, iterations is 12**

讨论收敛性：单点迭代是线性收敛的，由于在$[0,1]$中$\varphi'(x)=-e^{-x}/2,\left | \varphi'(x)\right | < 1/2,L=\frac{1}{2}$

## problem 2

$\varphi(x) = x + c(x^{2}-3),\varphi'(x) = 1+2cx$
$\varphi'(\alpha) = 1+2c\alpha$,$\alpha = \pm \sqrt3$
因此$\varphi'(\alpha) < 1$,即有$c\in(-\sqrt3,0)\cup(0,\sqrt 3)$

迭代速度用收敛阶计算，$\varphi'(\alpha) = 0,c = \pm \frac{1}{2\sqrt3}$

## problem 3

$x = \varphi(x) = \frac{1}{4}(\cos x+\sin x)$由于$\varphi'(x)<1$取$x\in(0,\frac{1}{2})$可以迭代

$x = \varphi(x) = 4 - 2^x$不可以迭代，需要改写

$x = \varphi_{1}(x) = \frac{\ln(4-x)}{\ln2}$在$[1,2]$迭代

$\varphi_1'(x) = \frac{1}{\ln2}\frac{1}{x-4}$在$x\in[1,2]$时$\varphi_1'(x)<1$收敛，可以迭代

## problem 4

$\varphi(x) = \frac{x(x^2+3a)}{3x^2+a}$

我们使用sympy库进行符号演算，结合$Taylor$展开

$\frac{\epsilon_{i+1}}{\epsilon_{i}^3}=\frac{1}{6}\varphi'''(\alpha)=\frac{1}{4a}$

## problem 5

$f(x) = (x^3 - 5)^2,f'(x) = 2(x^3 -5)* 3x^2$
有$\varphi(x) = x-\frac{f(x)}{f'(x)} = x - \frac{x^3 -5}{6x^2}$

由$Taylor$ 展开，这是个线性收敛,$c = \frac{1}{2}$

已知$r=2$,修改$x_{i+1} = x_i - r\frac{f(x_i)}{f'(x_i)}$

# Chapter 2

## problem 1

通过`Code_of_homeword/hw2/problem1.py`编程解决，并实现了Gauss消元的换行逻辑

## problem 2

通过`Code_of_homework/hw2/problem2.py`编程解决，实现了Gauss-Jordan消元，不考虑换行问题

## problem 3

通过`Code_of_homework/hw2/problem3.py`编程解决，实现了Doolittle消元和Crout消元

## problem 4

通过`Code_of_homework/hw2/problem4.py`编程解决，实现了Jacobi迭代和Gauss-Seidal迭代的过程，包括两种判断逻辑

## problem 5

本质是一个特征值计算的问题，第二问只要让特征值尽可能小，谱半径小$B^i$越小，迭代越快

另外最速降线和共轭梯度法在代码中

# Chapter 3

## problem 1

首先确定$\sum_{i=0}^{n}f(x_i)l_{i}(x)\in M_{n}$,接下来全部在$M_{n}$内讨论，可以视为对$M_{n}$内函数进行插值

$\sum_{i=0}^{n}l_i(x) -1$在$x_0...x_n$全为0，由代数学基本定理，$n$次多项式至多$n$个零点，得到$\sum_{i=0}^{n}l_i(x) =1$

$\sum_{i=0}^{n}x_il_i(x) -x$在$x_0...x_n$全为0，由代数学基本定理，$n$次多项式至多$n$个零点，得到$\sum_{i=0}^{n}l_i(x) =x$

## problem 2

