import sympy as sp

x,a = sp.symbols('x a')

f = (x**3 + 3*a*x) / (3*x**2 + a)

f_prime = sp.diff(f,x,3)

value = f_prime.subs({x**2:a})
print(value)