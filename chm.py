#!	Градиентный спуск + метод Ньютона, 1) 3)
#!	Дробление шага + Метод золотого сечения
#!	Андреев Артём Антонович 426

import matplotlib.pyplot as plt
import math
import random
import time
import numpy as np

#	Тестовые функции
def tfun1(a):
	x1 = a[0]
	x2 = a[1]
	return x1 ** 2 - x2 ** 2
def tfun2(a):
	x1 = a[0]
	x2 = a[1]
	return ((x1 - 1) ** 2 + (x2 + 1) ** 2)
def tfun3(a):
	x1 = a[0]
	x2 = a[1]
	x3 = a[2]
	return ((x1 - 1) ** 2 + (x2 + 1) ** 2 + (x3 - 4) ** 2)
def tfuna1(a):
	return 10 * (a[0] - 1) ** 2 + 0.1 * (a[1] - 1) ** 2 + (a[2] - 2) ** 2
def tfuna2(a):
	return np.exp(tfuna1(a))

#	Вычисление градиента
def gradient_f(f, v0, h):
	grad = np.zeros(len(v0))
	for i in range(len(v0)):
		v0[i] += h
		grad[i] = f(v0) / (2 * h)
		v0[i] -= 2 * h
		grad[i] -= f(v0) / (2 * h)
		v0[i] += h
	return grad

#	Матрциа Гёссе N - мерный случай
def gesse_matrix(f, v0, h):
	n = len(v0)
	e = np.eye(n)
	A = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			A[i][j] = (f(v0 + (e[i] + e[j]) * h) - f(v0 + e[j] * h) - f(v0 + e[i] * h) + f(v0)) / (h ** 2)
	return A

#	Шаг для метода Ньютона
def new_step(f, v0, h):
	A = np.linalg.inv(gesse_matrix(f, v0, h))
	return np.dot(A, gradient_f(f, v0, h))

#	Золотое сечение
def golden_ratio(f, v1, accuracy, hg):
	a = 0
	b = 10
	r = (3 - math.sqrt(5)) / 2
	c = a + r * (b - a)
	d = b - r * (b - a)
	while (b - a >= accuracy):
		if(f(v1 - hg * c) < f(v1 - hg * d)):
			b = d
			d = c
			c = a + r * (b - a)
		else:
			a = c
			c = d
			d = b - r * (b - a)
	return (a + b) / 2

def main(func):
####################################################
#	Установка начальных параметров + 1-й шаг
	eps, fx = 1.e-5, []
	accuracy = eps ** (1. / 2)
	h = 0.1 * accuracy
	beta = step = -0.001
	mu, la, bol = 2, 0.5, False
	n = int(input("Введите колличество аргументов функции\n"))
	v1 = np.array([float(input(f"Введите x{i + 1}\n")) for i in range(n)])
	grad = gradient_f(func, v1, h)
	v = v1 + grad * step
	fx.append(func(v))

####################################################
#	Градиентный спуск + Дробление Шага

	while (not bol):
		v1 = v
		grad = gradient_f(func, v, h)
		step = beta

		if (not (func(v + grad * step) < func(v))):
			step *= la
			if (not (func(v + grad * step) < func(v))):
				while (not (func(v + grad * step) < func(v))):
					step *= la
				step /= la
		else:
			step *= mu
			if (func(v + grad * step) < func(v + grad * beta)):
				while (func(v + grad * step) < func(v + grad * beta)):
					step *= mu
				step /= mu

		v = v1 + grad * step
		fx.append(func(v))
		bol = (abs(func(v) - func(v1)) <= accuracy) and (np.linalg.norm(v - v1) <= accuracy) and (np.linalg.norm(grad) <= accuracy)		
		

####################################################
#	Переход к методу Ньютона

	step = beta
	dot = func(v)
	i = len(fx)
	bol = False
	accuracy = accuracy ** 2

####################################################
#	Метод Ньютона + золотое сечение

	while (not bol):
		v1 = v
		grad = gradient_f(func, v, h)
		hg = new_step(func, v, h)
		step = golden_ratio(func, v1, accuracy, hg)
		v = v1 - hg * step
		fx.append(func(v))
		bol = (abs(func(v) - func(v1)) <= accuracy) and (np.linalg.norm(v - v1) <= accuracy) and (np.linalg.norm(grad) <= accuracy) 

####################################################
#	Вывод информации
	print(f"Номер итерации перехода на метод Ньютона: {i}\nЗначение при переходе : {dot}")
	print(f"Точка минимума : {v}")
	print(f"Значение функции: {func(v)}")
	fx = np.array(fx)
	plt.plot(range(1, len(fx) + 1), fx)
	plt.scatter(i, dot, c = 'r')
	plt.title(f"Обычный график. f(x) : i, где i = номер итерации")
	plt.show()
	plt.semilogy(range(1, len(fx) + 1), fx)
	plt.scatter(i, dot, c = 'r')
	plt.title(f"Логарифмическиг график. f(x) : i, где i = номер итерации")
	plt.show()

####################################################
try:
	main(tfuna2)
except Exception:
	print("Произошла ошибка в работе прораммы\nВероятно это связано с некоректно введёнными параметрами")