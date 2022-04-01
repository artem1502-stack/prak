import numpy as np
from numba import njit
import time
import math
import matplotlib.pyplot as plt
from copy import copy
from mpmath import *

"""

@njit
def f(x):
	return math.exp(-x)

@njit
def core(x, s):
	return 0.5 * x * math.exp(s)

@njit
def sres(a, b, n):
	x = line_devision(a, b, n)
	ua = np.zeros(n)
	for i in range(n):
		ua[i] = x[i] + math.exp(-x[i])
	return ua

"""

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
def bin_search(x, x0, m):
	n0 = 0
	nm = m
	i = m // 2
	if (x0 == x[-1]):
		return (len(x) - 1)
	elif (x0 == x[0]):
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
	m = len(u)
	i = bin_search(x, x0, m)		
	ua = u[i]
	ub = u[i + 1]
	p = (x0 - x[i]) / (x[i + 1] - x[i])
	res = ua + p * (ub - ua)
	return res

@njit
def line_devision(a, b, n):
	h = (b - a) / (n - 1)
	x = np.zeros(n)
	for i in range(n):
		x[i] = a + i * h
	return (x)

@njit
def core_matrix(a, b, n, c):
	x = line_devision(a, b, n)
	c = np.diag(c)
	K = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			K[i][j] = core(x[i], x[j])
	A = K.dot(c)
	res = A - np.eye(n)
	return (res)

@njit
def get_coeffs(a, b, n):
	x = np.zeros((n, n))
	#t = lambda x:
	X = line_devision(a, b, n)
	for i in range(n):
		for j in range(n):
			x[i][j] = (X[j]) ** i
	b = np.array([((b ** (k + 1)) - (a ** (k + 1))) / (k + 1) for k in range(n)])
	return np.linalg.solve(x, b)

@njit
def f_matrix(a, b, n):
	x = line_devision(a, b, n)
	F = np.zeros(n)
	for i in range(n):
		F[i] = -f(x[i])
	return F

@njit
def r_matrix(n, k):
	m = 1 + (n - 1) * k
	R = np.zeros((n, m))
	for i in range(n):
		R[i][i * k] = 1
	return (R)

#@njit
def solve_eq(a, b, n):
	c = get_coeffs(a, b, n)
	K = core_matrix(a, b, n, c)
	fm = f_matrix(a, b, n)
	u = np.linalg.solve(K, fm)
	return (u)

@njit
def l2_norm(u, u1, a, b):
	m = len(u1)
	x = line_devision(a, b, len(u))
	x1 = line_devision(a, b, len(u1))
	uh = np.zeros(m)
	for i in range(m):
		uh[i] = get_funx(u, x1[i], x)
	I = np.sum(np.power((uh - u1), 2))
	return (I)

@njit
def my_norm(u, u1, k = 2):
	mmax = 0
	m = len(u1)
	for i in range(m):
		if (i % k == 0):
			if abs(u[i // k] - u1[i]) > mmax:
				mmax = abs(u[i // k] - u1[i])
		elif abs(u1[i] - (u[i // k] + u[i // k + 1]) / 2) > mmax:
			mmax = abs(u1[i] - (u[i // k] + u[i // k + 1]) / 2)
	return (mmax)


#@njit
def solve_with_eps(a, b, eps):
	n = 4
	u = solve_eq(a, b, n)
	n *= 2
	u1 = solve_eq(a, b, n) 
	while (True):
		myn = my_norm(u, u1)
		l2n = l2_norm(u, u1, a, b)
		bo = (myn <= eps) and (l2n <= eps)
		if (bo):
			break
		u = u1.copy()
		n *= 2
		u1 = solve_eq(a, b, n)
	return u1

def main(a, b):
	eps = 1.e-6
	u = solve_with_eps(a, b, eps)
	n = len(u)
	ua = sres(a, b, n)
	x = line_devision(a, b, n)
	plt.title("Красная - численное решение, Синяя - аналитическое\n")
	plt.plot(x, u, "r", x, ua, "b")
	plt.axis([a, b, -10, 10])
	plt.show()
	with open("data.txt", "w") as f:
		print(*(u[::5] - ua[::5]), file = f)
	return(my_norm(ua, u))


if __name__ == "__main__":
	mp.dps = 20
	print(f"RES = {main(0, 2 * math.pi)}")


# pip install numpy numba matplotlib