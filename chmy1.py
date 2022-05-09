import numpy
import math
import time
from numba import njit

@njit
def test_func(args):
    x = args[0]
    y = args[1]
    return (x - 3) ** 2 + (y + 1) ** 2

@njit
def gradient(fun, x, h):
    grad = numpy.zeros(len(x))
    for i in range(len(x)):
        x[i] += h
        grad[i] = fun(x) / (2 * h)
        x[i] -= 2 * h
        grad[i] -= fun(x) / (2 * h)
        x[i] += h
    return grad

def step_func(arg, alpha, f, h):
    res = x - gradient(f, x, h) * alpha
    return (res)
#F(alfa) = f(x - grad * alfa)
@njit
def der(f, x, h, grad, step):
    res = f(x - grad * (step + h)) - f(x - grad * (step - h)) / 2 * h
    return (res)

@njit
def tangent_method(f, x, accur, grad, h):
    a = 0.01
    b = 100
    f_a = f(x - grad * a)#num
    f_b = f(x - grad * b)#num
    der_a = der(f, x, h, grad, a)
    der_b = der(f, x, h, grad, b)
    if (der_a > 0 and der_b > 0):
        return a
    if (der_a < 0 and der_b < 0):
        return b
    c = (f_a - f_b) / (der_b - der_a) + (der_b * b - der_a * a) / (der_b - der_a)
    #c = c / (der_b - der_a)
    der_c = der(f, x, h, grad, c)
    while abs(der_c) > accur:
        if der_c <= 0:
            a = c
        elif der_c > 0:
            b = c
        f_a = f(x - grad * a)  # num
        f_b = f(x - grad * b)  # num
        der_a = der(f, x, h, grad, a)
        der_b = der(f, x, h, grad, b)
        c = (f_a - f_b) / (der_b - der_a) + (der_b * b - der_a * a) / (der_b - der_a)
        #c = c / (der_b - der_a)
        der_c = der(f, x, h, grad, c)
    return (c)

def do_grad_method(f, h, x, accur):
    grad = gradient(f, x, h)
    alpha = tangent_method(f, x, accur, grad, h)
    return (x - grad * alpha), grad


if __name__ == '__main__':
    res = list()
    f = test_func
    eps = 10 ** (-9)
    accur = math.sqrt(eps)
    h = 0.1 * accur
    beta = alpha = 0.001
    MU, LAMBDA = 2, 0.5
    n = 2
    x1 = numpy.array([float(input("Введите аргумент номер {0}\n".format(i))) for i in range(n)])
    grad = gradient(f, x1, h)
    x = x1 - grad * alpha
    res.append(f(x))
    while abs(f(x) - f(x1)) > accur or numpy.linalg.norm(x - x1) > accur or numpy.linalg.norm(grad) > accur:
        #print(numpy.linalg.norm(grad))
        x1 = x
        x, grad = do_grad_method(f, h, x, accur)
        res.append(f(x))
        #time.sleep(0.1)
    print("OKOKOK")
    print(f"{res[-1]}")