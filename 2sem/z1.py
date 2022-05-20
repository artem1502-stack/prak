from numba import njit
import numba
import numpy as np
import matplotlib.pyplot as plt
import time

eps = 1.e-9

@njit
def test_function(x : float, y : float) -> float:
	return (2 * x)

class stepOne:

	cname = 'stepOne'
	p = 2
	@staticmethod
	@njit
	def solve(step, x, y0):
		y = x * 0
		y[0] = y0
		n = 0
		for i in range(len(x) - 1):
			k1 = step[i] * test_function(x[i], y[i])
			k2 = step[i] * test_function(x[i] + step[i], y[i] + k1)
			y[i + 1] = y[i] + 0.5 * (k1 + k2)
			n += 2
		return y, n

	@staticmethod
	@njit
	def next(step, x, y):
		k1 = step * test_function(x, y)
		k2 = step * test_function(x + step, y + k1)
		return y + 0.5 * (k1 + k2)

	@staticmethod
	@njit
	def error(m1, m2, s):
		m3, _ = half_split(m1)
		return np.max(np.abs((m3 - m2))) /  (2 ** s - 1)

class stepTwo:

	cname = 'stepTwo'
	p = 3
	@staticmethod
	@njit
	def solve(step, x, y0):
		y = x * 0
		y[0] = y0
		n = 0
		for i in range(len(x) - 1):
			k1 = step[i] * test_function(x[i], y[i])
			k2 = step[i] * test_function(x[i] + step[i] * 0.5, y[i] + 0.5 * k1)
			k3 = step[i] * test_function(x[i] + step[i], y[i] - k1 + 2 * k2)
			y[i + 1] = y[i] + (1/6.) * (k1 + 4 * k2 + k3)
			n += 3
		return y, n

	@staticmethod
	@njit
	def error(m1, m2, s):
		m3, _ = half_split(m1)
		return np.max(np.abs((m3 - m2))) /  (2 ** s - 1)

class stepThree:


	cname = 'stepThree'
	p = 3
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
			e[i] = (2/3.) * (k1 - k2 - k3 + k4)
			y[i + 1] = y[i] + (1/6.) * (k1 + 2 * k2 + 2 * k3 + k4)
		k1 = step[i] * test_function(x[i], y[i])
		k2 = step[i] * test_function(x[i] + step[i] * 0.5, y[i] + 0.5 * k1)
		k3 = step[i] * test_function(x[i] + step[i] * 0.5, y[i] + 0.5 * k2)
		k4 = step[i] * test_function(x[i] + step[i], y[i] + k3)
		e[i] = (2/3.) * (k1 - k2 - k3 + k4)
		return y, n, e

	@staticmethod
	@njit
	def error(e1, e2, s):
		e3, _ = half_split(e1)
		return np.max(np.abs((e3 - e2)))




def rand_split_line(a, b, n):
	mas = np.sort(np.random.uniform(a, b, n))
	delta_mas = np.array([mas[i + 1] - mas[i] for i	in range(len(mas) - 1)])
	return mas, delta_mas

def split_line(a, b, n):
	pass


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
def control_error():
	pass

def progress_bar(n_iter, norm, eps, n):
	print(f"\r|n_iter = {n_iter:4}, norm = {norm:1.3e}, eps/norm% = {((eps / norm) * 100):2.3f}%, n_points = {n:5}|", end="\r")


def new_progress_bar(n, mn, mx):

	print(f"\r|currently {n} points in arr. step is from {mn} to {mx}|", end='\r')

def do_step(step_class, a, b, y0, n = 100, s = 5, graph = False, progress = True, e = False):
	x1, delta1 = split_line(a, b, n)
	x2, delta2 = half_split(x1, delta1)
	f = test_function
	print(step_class.cname)
	func_uses = 0
	if e:
		y1, temp_n, e1 = step_class.solve(delta1, x1, y0)
		func_uses += temp_n
		y2, temp_n, e2 = step_class.solve(delta2, x2, y0)
		func_uses += temp_n
		norm = step_class.error(e1, e2, s)
	else:
		y1, temp_n = step_class.solve(delta1, x1, y0)
		func_uses += temp_n
		y2, temp_n = step_class.solve(delta2, x2, y0)
		func_uses += temp_n
		norm = step_class.error(y1, y2, s)
	i = 0
	while (norm > eps):
		if progress: progress_bar(i, norm, eps, len(x2))
		y1, x1, delta1= y2, x2, delta2
		if (e):
			e1 = e2
		x2, delta2 = half_split(x1, delta1)
		if e:
			y2, temp_n, e2 = step_class.solve(delta2, x2, y0)
		else:
			y2, temp_n = step_class.solve(delta2, x2, y0)
		func_uses += temp_n
		i += 1
		if e: norm = step_class.error(e1, e2, s)
		else: norm = step_class.error(y1, y2, s)
	
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

def full(a, b, y0, n = 100, s = 5, graph = False):
	t = time.time()
	x1, y1 = do_step(stepOne, a, b, y0, n, 2)
	x2, y2 = do_step(stepTwo, a, b, y0, n, 3)
	x3, y3 = do_step(stepThree, a, b, y0, n, 3, e = True)
	print(f"Time: {time.time() - t :.3}s")
	for xt, yt in zip([x1,x2,x3], [y1, y2, y3]):	
		y = np.exp(xt)
		md = np.abs(yt - y)
		mi = md.argmax()
		print(f"accuracy = {md[mi] / y[mi] * 100}%")
	if graph:
		form = [len(i) // 10000 + 1 for i in [x1,x2,x3]]
		plt.plot(x1[::form[0]], y1[::form[0]], 'r', x2[::form[1]], y2[::form[1]],'b',\
				x3[::form[2]], y3[::form[2]], 'c', x3[::form[2]], y[::form[2]], 'g')
		plt.show()


def new_solve(cls, n, a, b, y0):
	x = a
	y = y0
	x_list = [x]
	y_list = [y0]
	p = cls.p
	step = (b - a) / (n - 1)
	min_step = step
	max_step = step
	ln = 1
	while (x < b):
		y1 = cls.next(step, x, y)
		y2_1 = cls.next(step / 2, x, y)
		y2 = cls.next(step / 2, x + step / 2, y2_1)
		while (np.abs((y2 - y1) / (1 - 2 ** (-p))) > (eps * step) / (b - a)):
			step /= 2
			y1 = cls.next(step, x, y)
			y2_1 = cls.next(step / 2, x, y)
			y2 = cls.next(step / 2, x + step / 2, y2_1)
			ln += 1
		x = x + step
		y = y1
		x_list.append(x)
		y_list.append(y)
		ln += 1
		max_step = max(step, max_step)
		min_step = min(step, min_step)
		new_progress_bar(ln, min_step, max_step)
		if (x + 2 * step < b): step *= 2
	return np.array(x_list), np.array(y_list)

x,y = new_solve(stepOne, 100, 0, 1, 0)
yt = x ** 2
print('\n')
print(len(x))
print(np.linalg.norm(np.abs(y - yt)))
plt.plot(x, y, 'r', x, yt, 'b')
plt.show()

