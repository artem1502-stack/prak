from numba import njit

class stepOne:
	
	@staticmethod
	@njit
	def k1(step, function, x, y):
		return step * function(x, y)

	@staticmethod
	@njit
	def k2(step, function, x, y):
		return step * function(x + step, y + stepOne.k1(step, function, x, y))

	@staticmethod
	@njit
	def solve(step, function, x, y0):
		y = x * 0
		y[0] = y0
		for i in range(len(x) - 1):
			y[i + 1] = y[i] + 0.5 * (stepOne.k1(step[i], function, x[i], y[i])\
									+ stepOne.k2(step[i], function, x[i], y[i]))
		return y

class stepTwo:

	@classmethod
	def k1(cls, step, function, x, y):
		return step * function(x, y)

	
	@classmethod
	def k2(cls, step, function, x, y):
		return step * function(x + step * 0.5, y + 0.5 * cls.k1(step, function, x, y))

	@classmethod
	def k3(cls, step, function, x, y):
		return step * function(x + step , y - cls.k1(step, function, x, y) + 2 * cls.k2(step, function, x, y))

	@classmethod
	def solve(cls, step, function, x, y0):
		y = x * 0
		y[0] = y0
		for i in range(len(x) - 1):
			y[i + 1] = y[i] + (1/6.) * (cls.k1(step[i], function, x[i], y[i])\
									+ 4 * cls.k2(step[i], function, x[i], y[i])\
									+ cls.k3(step[i], function, x[i], y[i]))
		return y
