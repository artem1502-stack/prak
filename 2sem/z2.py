from sympy import solve, dsolve, Eq, symbols, Function, Matrix


x = Function("x")
t = symbols("t")

print(dsolve(Eq(x(t).diff(t), x(t)), x(t)))

print(Matrix([[1, 2], [3, 4]]).eigenvals())
