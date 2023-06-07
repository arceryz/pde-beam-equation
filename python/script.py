#!/bin/python
import scipy
from numpy import cos, sin, cosh, sinh, exp
import scipy.optimize as optimize
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
from decimal import *

#################################################################################
# Settings.
#################################################################################

L = 200
motes = 20
rootfinder_method = "brentq"


#################################################################################
# Computing the eigenvalues.
#################################################################################

eigenvalues = np.zeros(motes)

def ev_function(x) -> float:
    return cosh(x) * cos(x) + 1

def compute_evs():
    r = 0.3043077 * 1.1
    for i in range(0, motes):
        c = scipy.pi * (i + 0.5)

        # Compute the root in normalized coordinates then scale back.
        yr = optimize.root_scalar(ev_function, bracket=[c-r, c+r],
                                  method=rootfinder_method).root
        xr = yr/L
        eigenvalues[i] = xr

compute_evs()
#print(eigenvalues)


#################################################################################
# Computing the eigenfunctions.
#################################################################################

def phi_eigenfunc_bad(x, n):
    # Numerically WORSE variant.
    ev = eigenvalues[n]
    c = (cosh(ev*L) + cos(ev*L)) / \
        (sinh(ev*L) + sin(ev*L))
    y = cosh(ev*x) - cos(ev*x) - c * (sinh(ev*x) - sin(ev*x))
    return y

def phi_eigenfunc(x, n):
    ev = eigenvalues[n]
    c = (cosh(ev*L) + cos(ev*L)) / \
        (sinh(ev*L) + sin(ev*L))
    y = ((1-c)*exp(ev*x) + (1+c)*exp(-ev*x))/2.0 + c*sin(ev*x) - cos(ev*x)
    return y

def plot_eigenfunc(n):
    pts = 999
    xlist = np.linspace(0, L, pts)
    ylist = np.zeros(pts)

    for i in range(pts):
        ylist[i] = phi_eigenfunc(xlist[i], n)
    plt.plot(xlist, ylist)

def plot_eigenfuncs():
    for n in range(3):
        plot_eigenfunc(n)
    plt.title("First three motes of eigenfunctions")
    plt.xlabel("Position (x)")
    plt.ylabel("Y")
    plt.xlim(0, L)
    plt.show()

def plot_eigenfuncs_high():
    plot_eigenfunc(motes-1)
    plt.title("Highest motes of eigenfunctions")
    plt.xlabel("Position (x)")
    plt.ylabel("Y")
    plt.xlim(0, L)
    plt.show()

#plot_eigenfuncs_high()


#################################################################################
# Inner product <phi, phi> numerically.
#
# We research here what happens if we integrate phi^2.
# This is needed for further computations.
#################################################################################

def phi_inner_num(n):
    return integrate.quad(lambda x: phi_eigenfunc(x,n)**2, 0, L)[0]

def phi_inner(n):
    return L

def test_phi_inner():
    for n in range(0, 10):
     print(phi_inner_num(n))

#test_phi_inner():

# We conclude from these prints that the integral <phi,phi> = L. Thus we skip
# computing this integral and use it without proof.


#################################################################################
# Fourier coefficients computation.
#
# In this section we compute the integrals for fourier coefficients of
# arbitrary functions. We test this with some sample functions.  
# Then we store the coefficients we need in lists for later use.
#################################################################################

def fourier_coeffs_arr(f, n) -> np.array:
    coeffs = np.zeros(n)
    for i in range(n):
        y = integrate.quad(lambda x: f(x)*phi_eigenfunc(x, i), 0, L)[0]
        coeffs[i] = y
    return coeffs

def fourier_eval(x, coeffs) -> float:
    y = 0 
    for i in range(len(coeffs)):
        y += coeffs[i] * phi_eigenfunc(x, i)
    return y

def plot_fourier_test():
    pts = 999
    xlist = np.linspace(0, L, pts)
    ylist = np.zeros(pts)

    coeffs = fourier_coeffs_arr(lambda x: 0.01*x, 10)
    for i in range(pts):
        ylist[i] = fourier_eval(xlist[i], coeffs)
    plt.plot(xlist, ylist)
    plt.show()

plot_fourier_test()
