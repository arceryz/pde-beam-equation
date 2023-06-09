#!/bin/python
import scipy.integrate as integrate
import numpy as np

def f(x):
    if x > 5:
        return 1
    else:
        return 0

def g(xlist):
    return list(map(f, xlist))

print(integrate.quadrature(g, 0, 10)[0])
