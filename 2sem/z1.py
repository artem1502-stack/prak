from numba import njit
import numba
import numpy as np
import matplotlib.pyplot as plt
import time

@njit
def test_function(x : float, y : float) -> float:
	return (y)

class stepOne:

	@staticmethod
	@njit
	def solve(step, x, y0):
		y = x * 0
		y[0] = y0
		for i in range(len(x) - 1):
			k1 = step[i] * test_function(x[i], y[i])
			k2 = step[i] * test_function(x[i] + step[i], y[i] + k1)
			y[i + 1] = y[i] + 0.5 * (k1 + k2)
		return y
class stepTwo:

	@staticmethod
	@njit
	def solve(step, x, y0):
		y = x * 0
		y[0] = y0
		for i in range(len(x) - 1):
			k1 = step[i] * test_function(x[i], y[i])
			k2 = step[i] * test_function(x[i] + step[i] * 0.5, y[i] + 0.5 * k1)
			k3 = step[i] * test_function(x[i] + step[i], y[i] - k1 + 2 * k2)
			y[i + 1] = y[i] + (1/6.) * (k1 + 4 * k2 + k3)
		return y
"""
class stepThree:

	@staticmethod
	@njit
	def solve(step, x, y0):
		y = x * 0
		e = x * 0 
		y[0] = y0
		for i in range(len(x) - 1):
			k1 = step[i] * test_function(x[i], y[i])
			k2 = step[i] * test_function(x[i] + step[i] * 0.5, y[i] + 0.5 * k1)
			k3 = step[i] * test_function(x[i] + step[i] * 0.5, y[i] + 0.5 * k2)
			k4 = step[i] * test_function(x[i] + step[i], y[i] + k3)
			e[i] = (2/3.)(k1 - k2 - k3 + k4)
			y[i + 1] = y[i] + (1/6.) * (k1 + 2 * k2 + 2 * k3 + k4)
		k1 = step[i] * test_function(x[i], y[i])
		k2 = step[i] * test_function(x[i] + step[i] * 0.5, y[i] + 0.5 * k1)
		k3 = step[i] * test_function(x[i] + step[i] * 0.5, y[i] + 0.5 * k2)
		k4 = step[i] * test_function(x[i] + step[i], y[i] + k3)
		e[i] = (2/3.)(k1 - k2 - k3 + k4)
		return y, e
"""
@njit
def split_line(a, b, n):
	mas = np.sort(np.random.uniform(a, b, n))
	delta_mas = np.array([mas[i + 1] - mas[i] for i	in range(len(mas) - 1)])
	return mas, delta_mas
@njit
def half_split(mas, delta_mas = None):
	n = len(mas) * 2 - 1
	n_mas = np.array([mas[i//2] if (i % 2 == 0) else (mas[i // 2] + mas[i // 2 + 1])/ 2. for i in range(n)])
	if delta_mas is None:
		return n_mas, None
	n_delta = np.array([delta_mas[i // 2] / 2. for i in range(n - 1)])
	return n_mas, n_delta
def print_mas(*mmas):
	for mas in mmas:
		for i in mas:
			print(f"{i:.4} ", end='')
		print('')
@njit
def error(m1, m2, s):
	m3, _ = half_split(m1)
	return np.max(np.abs((m3 - m2))) /  (2 ** s - 1)
@njit
def control_error():
	pass
def progress_bar(n_iter, norm, eps, n):
	print(f"\r|n_iter = {n_iter:4}, norm = {norm:1.3e}, eps/norm% = {((eps / norm) * 100):2.3f}%, n_points = {n:5}|", end="\r")
def do_step(step_class, a, b, y0, n = 100, s = 5, graph = False, progress = True):
	eps = 1.e-9
	x1, delta1 = split_line(a, b, n)
	x2, delta2 = half_split(x1, delta1)
	f = test_function
	y1 = step_class.solve(delta1, x1, y0)
	y2 = step_class.solve(delta2, x2, y0)
	norm = error(y1, y2, s)
	i = 0
	while (norm > eps):
		if progress: progress_bar(i, norm, eps, len(x2))
		y1, x1, delta1= y2, x2, delta2
		x2, delta2 = half_split(x1, delta1)
		y2 = step_class.solve(delta2, x2, y0)
		i += 1
		norm = error(y1, y2, s)
	if progress:
		progress_bar(i, norm, eps, len(x2))
		print('')
	if graph:
		y = np.exp(x2)
		plt.plot(x2, y2, 'r', x2, y, 'b')
		plt.show()
	return x2, y2

def step_one(a, b, y0, n = 100, s = 5):
	do_step(stepOne, a, b, y0, n, s, graph = True)


def full(a, b, y0, n = 100, s = 5):
	t = time.time()
	x1, y1 = do_step(stepOne, a, b, y0, n, 2)
	x2, y2 = do_step(stepTwo, a, b, y0, n, 3)
	print(f"Time: {time.time() - t :.3}s")
	y = np.exp(x2)
	form = len(x2) // 10000 + 1
	plt.plot(x1[::form], y1[::form], 'r', x2[::form], y2[::form], 'b', x2[::form], y[::form], 'g')
	plt.show()

full(0, 5, 1)
