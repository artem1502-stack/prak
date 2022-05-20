from numba import njit
import numba
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint

eps = 1.e-6

@njit
def test_function(x : float, y : float) -> float:
	return (2*x)

@njit
def tf_forscipy(y : float, x : float) -> float:
	return (2*x)


class stepOne:

	cname = 'stepOne'
	p = 2

	@staticmethod
	@njit
	def next(step, x, y):
		k1 = step * test_function(x, y)
		k2 = step * test_function(x + step, y + k1)
		return y + 0.5 * (k1 + k2), 2

	@classmethod
	def solve(cls, y0, a = 0, b = 1, n = 128):
		print(cls.cname)
		x = a
		y = y0
		x_list = [x]
		y_list = [y0]
		p = cls.p
		step = (b - a) / (n - 1)
		min_step = step
		max_step = step
		ln = 1
		f_calls = 0
		while (x < b):
			y1, temp_n = cls.next(step, x, y)
			f_calls += temp_n
			y2_1, temp_n = cls.next(step / 2, x, y)
			f_calls += temp_n
			y2, temp_n = cls.next(step / 2, x + step / 2, y2_1)
			f_calls += temp_n
			while (np.abs((y2 - y1) / (1 - 2 ** (-p))) >= (eps * step) / (b - a)):
				step /= 2
				y1, temp_n = cls.next(step, x, y)
				f_calls += temp_n
				y2_1, temp_n = cls.next(step / 2, x, y)
				f_calls += temp_n
				y2, temp_n = cls.next(step / 2, x + step / 2, y2_1)
				f_calls += temp_n
			x = x + step
			y = y1
			if (x <= b):
				x_list.append(x)
				y_list.append(y)
			ln += 1
			max_step = max(step, max_step)
			min_step = min(step, min_step)
			new_progress_bar(ln, min_step, max_step, f_calls, x / (b - a) * 100)
			step *= 2
		print('')
		return np.array(x_list), np.array(y_list), f_calls

class stepTwo:

	cname = 'stepTwo'
	p = 3

	@staticmethod
	@njit
	def next(step, x, y):
		k1 = step * test_function(x, y)
		k2 = step * test_function(x + step * 0.5, y + 0.5 * k1)
		k3 = step * test_function(x + step, y - k1 + 2 * k2)
		return y + (1/6.) * (k1 + 4 * k2 + k3), 3

	@classmethod
	def solve(cls, dcls, y0, a = 0, b = 1, n = 128):
		print(cls.cname)
		x = a
		y = y0
		x_list = [x]
		y_list = [y0]
		p = cls.p
		step = (b - a) / (n - 1)
		min_step = step
		max_step = step
		f_calls = 0
		ln = 1
		while (x < b):
			y1, temp_n = dcls.next(step, x, y)
			f_calls += temp_n
			y2, temp_n = cls.next(step, x, y)
			f_calls += temp_n
			while (np.abs(y2 - y1) >= (eps * step) / (b - a)):
				step /= 2
				y1, temp_n = dcls.next(step, x, y)
				f_calls += temp_n
				y2, temp_n = cls.next(step, x, y)
				f_calls += temp_n
			x = x + step
			y = y1
			if (x <= b):
				x_list.append(x)
				y_list.append(y)
			ln += 1
			max_step = max(step, max_step)
			min_step = min(step, min_step)
			new_progress_bar(ln, min_step, max_step, f_calls, (x - a) / (b - a) * 100)
			step *= 2
		print('')
		return np.array(x_list), np.array(y_list), f_calls

class stepThree:

	cname = 'stepThree'
	p = 3

	@staticmethod
	@njit
	def next(step, x, y):
		k1 = step * test_function(x, y)
		k2 = step * test_function(x + step * 0.5, y + 0.5 * k1)
		k3 = step * test_function(x + step * 0.5, y + 0.5 * k2)
		k4 = step * test_function(x + step, y + k3)
		e = (2/3.) * (k1 - k2 - k3 + k4)
		return y + (1/6.) * (k1 + 4 * k2 + k3), 3, np.abs(e)


	@classmethod
	def solve(cls, y0, a = 0, b = 1, n = 128):
		print(cls.cname)
		x = a
		y = y0
		x_list = [x]
		y_list = [y0]
		p = cls.p
		step = (b - a) / (n - 1)
		min_step = step
		max_step = step
		f_calls = 0
		ln = 1
		while (x < b):
			y1, temp_n, e = cls.next(step, x, y)
			f_calls += temp_n
			while (e >= (eps * step) / (b - a)):
				step /= 2
				y1, temp_n, e = cls.next(step, x, y)
				f_calls += temp_n
			x = x + step
			y = y1
			if (x <= b):
				x_list.append(x)
				y_list.append(y)
			ln += 1
			max_step = max(step, max_step)
			min_step = min(step, min_step)
			new_progress_bar(ln, min_step, max_step, f_calls, (x - a) / (b - a) * 100)
			step *= 2
		print('')
		return np.array(x_list), np.array(y_list), f_calls


def new_progress_bar(n, mn, mx, f_calls, prc):
	if prc > 100:
		prc = 100
	print(f"\r|step in [{mn:.4e}, {mx:.4e}], f_calls={f_calls}. {prc:.4f}% completed|", end='\r')



def main(y0, a, b, n = 128, graph = False):

	x1, y1, f_calls = stepOne.solve(y0, a, b, n)
	print(f"function was called {f_calls} times")
	print(f"used {len(x1)} points")
	x2, y2, f_calls = stepTwo.solve(stepOne, y0, a, b, n)
	print(f"function was called {f_calls} times")
	print(f"used {len(x2)} points")
	x3, y3, f_calls = stepThree.solve(y0, a, b, n)
	print(f"function was called {f_calls} times")
	print(f"used {len(x3)} points")
	
	"""
	with open("res.txt", "w") as f:
		print(*x1, file=f)
		print(*x2, file=f)
		print(*x3, file=f)
	"""


	fsolve = np.vectorize(lambda x: x**2)

	x = max(x1, x2, x3, key=len)
	y_py = odeint(tf_forscipy, y0, x)
	yt = fsolve(x)
	print(y_py)	


	print(f"accuracy on stepOne = {np.max(np.abs(y1 - fsolve(x1))):.4e}")
	print(f"accuracy on stepTwo = {np.max(np.abs(y2 - fsolve(x2))):.4e}")
	print(f"accuracy on stepThree = {np.max(np.abs(y3 - fsolve(x3))):.4e}")
	print(f"accuracy on scipy = {np.max(np.abs(y_py[:,0] - yt)):.4e}")
	if graph:
		plt.plot(x1, y1, 'r', x2, y2, 'g', x3, y3, 'b', x, yt, 'c')
		plt.show()

main(0, 0, 1)
