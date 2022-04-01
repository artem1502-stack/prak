import numpy as np
from numba import njit
import time
import math
import matplotlib.pyplot as plt
from copy import copy
@njit
def f(x):
	return math.cos(2 * x)
@njit
def core(x, s):
	return math.sin(x) * math.cos(s)
@njit
def sres(a, b, n):
	x = line_devision(a, b, n)
	ua = np.zeros(n)
	for i in range(n):
		ua[i] = math.cos(2 * x[i])
	return ua
@njit
def line_devision(a, b, n):
	h = (b - a) / (n - 1)
	x = np.zeros(n)
	for i in range(n):
		x[i] = a + i * h
	return (x)
@njit
def r_matrix(n):
	R = [np.zeros((n - 1, 2 * n - 1)) for _ in range(3)]
	for i in range(n - 1):
		R[0][i][2 * i] = 1
		R[1][i][2 * i + 2] = 1
		R[2][i][2 * i + 1] = 1
	res = R[0].transpose().dot(R[0])\
		+ R[1].transpose().dot(R[1])\
		+ 4 * R[2].transpose().dot(R[2])
	return (res)
@njit
def k_matrix(a, b, n):
	m = 2 * n - 1
	x = line_devision(a, b, m)
	K = np.zeros((m, m))
	for i in range(m):
		for j in range(m):
			K[i][j] = core(x[i], x[j])
	return (K)
@njit
def f_matrix(a, b, n):
	m = 2 * n - 1
	x = line_devision(a, b, m)
	F = np.zeros(m)
	for i in range(m):
		F[i] = f(x[i])
	return (F)
@njit
def solve_eq(a, b, n):
	m = 2 * n - 1
	h = (b - a) / (n - 1)
	A = np.eye(m) - (h / 6) * k_matrix(a, b, n).dot(r_matrix(n))
	fm = f_matrix(a, b, n)  
	u = np.linalg.solve(A, fm)
	return (u)
@njit
def bin_search(x, x0, m):
	n0 = 0
	nm = m - 1
	i = n0 + (nm - n0) // 2
	if (x0 >= x[-1] and x0 <= x[-1]):
		return (len(x) - 2)
	elif (x0 >= x[0] and x0 <= x[0]):
		return (0)
	while (True):
		if (x[i] <= x0) and (x0 <= x[i + 1]):
			break
		if x0 < x[i]:
			nm = i
		else:
			n0 = i
		i = n0 + (nm - n0) // 2
	return i
@njit
def get_funx(u, x0, x):
	assert x0 <= x[-1] and x0 >= x[0]
	m = len(u)
	i = bin_search(x, x0, m)		
	ua = u[i]
	ub = u[i + 1]
	p = (x0 - x[i]) / (x[i + 1] - x[i])
	res = ua + p * (ub - ua)
	return res
@njit
def norm_s(u, u1, a, b):
	m = len(u1)
	x = line_devision(a, b, len(u))
	x1 = line_devision(a, b, len(u1))
	uh = np.zeros(m)
	for i in range(m):
		uh[i] = get_funx(u, x1[i], x)
	d = uh - u1
	I1 = np.sum(np.power(d, 2))
	I2 = max(np.abs(d))
	return I1, I2
@njit
def solve_with_eps(a, b, eps):
	n = 4
	u = solve_eq(a, b, n)
	n *= 2
	u1 = solve_eq(a, b, n)
	norm, _ = norm_s(u, u1, a, b)
	while (norm > eps):
		u = u1.copy()
		n *= 2
		u1 = solve_eq(a, b, n)
		norm, _ = norm_s(u, u1, a, b)
	return u1

def main():
	t = time.time()
	eps = 1.e-6
	a = 0
	b = 2 * math.pi
	u = solve_with_eps(a, b, eps)
	print(f"Время расчётов: {round(time.time() - t, 3)} секунд")
	n = len(u)
	ua = sres(a, b, n)
	print(f"На n = {n} точек, нормы (L2, C) = {tuple(norm_s(u, ua, a, b))}")
	x = line_devision(a, b, n)
	plt.title("Красная - численное решение, Синяя - аналитическое\n")
	plt.plot(x, u, "r", x, ua, "b")
	plt.axis([a, b, 1.5 * min(ua), 1.5 * max(ua)])
	plt.show()
	with open("data.txt", "w") as f:
		print(*(u[::5] - ua[::5]), file = f)

if __name__ == "__main__":
	main()