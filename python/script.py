#!/bin/python
import scipy
import scipy.optimize as optimize
import scipy.integrate as integrate
from scipy.misc import derivative
import numpy as np
from numpy import cos, sin, cosh, sinh, exp
import matplotlib.pyplot as plt

from p_tqdm import p_map


#################################################################################
# Settings.
#################################################################################

L = 200
E = 1 
I = 1
mu = 1

motes = 10
rootfinder = "brentq"
dx = 1e-6
quad_tolerance_lowp = 1e-4
quad_tolerance_highp = 1.49e-8

def forcing(x, t):
    return 1

def ic_deflection(x):
    return 0.01*x

def ic_velocity(x):
    return 0


#################################################################################
# Tools.
#################################################################################

def diff(f, x, dx):
    return (f(x) + f(x+dx))/dx

def integral_lowp(f, a, b):
#    return integrate.quadrature(f, a, b, tol=quad_tolerance_lowp)[0]
    return integrate.quad(f, a, b)[0]

def integral_highp(f, a, b):
    #return integrate.quadrature(f, a, b, tol=quad_tolerance_highp)[0]
    return integrate.quad(f, a, b)[0]


#################################################################################
# Computing the eigenvalues.
#################################################################################

eigenvalues = np.zeros(motes)
alfas = np.zeros(motes)

def ev_function(x) -> float:
    return cosh(x) * cos(x) + 1

def compute_evs():
    r = 0.3043077 * 1.1
    for i in range(0, motes):
        c = scipy.pi * (i + 0.5)

        # Compute the root in normalized coordinates then scale back.
        yr = optimize.root_scalar(ev_function, bracket=[c-r, c+r],
                                  method=rootfinder).root
        xr = yr/L
        eigenvalues[i] = xr
        alfas[i] = np.sqrt(xr*E*I / mu)

compute_evs()
#print("Eigenvalues %s" % str(eigenvalues))
#print("Alfas %s" % str(alfas))


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

#plot_eigenfuncs()
#plot_eigenfuncs_high()


#################################################################################
# Inner product <phi, phi> numerically.
#
# We research here what happens if we integrate phi^2.
# This is needed for further computations.
#################################################################################

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

def fourier_coeff(f, n) -> float:
    y = integral_lowp(lambda x: f(x)*phi_eigenfunc(x, n), 0, L) / L
    return y

def fourier_coeffs_arr(f, n) -> np.array:
    coeffs = np.zeros(n)
    for i in range(n):
        coeffs[i] = integral_highp(lambda x: f(x)*phi_eigenfunc(x, i), 0, L) / L
    return coeffs

def fourier_eval(x, coeffs) -> float:
    y = 0 
    for i in range(len(coeffs)):
        y += coeffs[i] * phi_eigenfunc(x, i)
    return y

def plot_fourier_series(testfun):
    pts = 999
    xlist = np.linspace(0, L, pts)
    ylist = np.zeros(pts)
    ylist2 = np.zeros(pts)

    coeffs = fourier_coeffs_arr(testfun, motes)
    for i in range(pts):
        ylist[i] = fourier_eval(xlist[i], coeffs)
        ylist2[i] = testfun(xlist[i])
    plt.plot(xlist, ylist)
    plt.plot(xlist, ylist2)

#plot_fourier_series(lambda x: (x % 100.0) / 100.0)
#plt.show()

fourier_defl = fourier_coeffs_arr(ic_deflection, motes)
fourier_vel  = fourier_coeffs_arr(ic_velocity, motes)
#print("Deflection fourier coeffs: %s" % str(fourier_defl))
#print("Velocity fourier coeffs: %s" % str(fourier_vel))


#################################################################################
# Time 1: Particular solution computation.
#
# This section computes the particular solution for the time coefficients.
# This is a function obtained by variation of parameters.
# By far the most complicated thing to solve for.
#################################################################################

def fourier_force(t, n):
    # This defines the function Fhat(t), the fourier coefficient of the 
    # forcing at a single time t.
    # We have to use a lambda x to make the forcing a function of just x.
    y = fourier_coeff(lambda x: forcing(x, t), n)
    return y


def psi_particular(t, n):
    # A much nicer form of the integral is after some clever algebra.
    # It yields this expression where t is mixed in the integral.
    alfa = alfas[n]
    y = 1/alfa*integral_lowp(lambda z: 
                fourier_force(z, n)*sin(alfa*(t-z)),0,t)
    return y

def plot_psi(n):
    pts = 100
    tlist = np.linspace(0, 50, pts)
    ylist = np.zeros(pts)
    ylist2 = np.zeros(pts)

    for i in range(pts):
        ylist[i] = psi_particular(tlist[i],n)
    plt.plot(tlist, ylist)

#plot_psi(0)
#plot_psi(1)
#plot_psi(2)
#plt.show()


#################################################################################
# Time 2: Time coefficients computation.
#
# We now need to find the time coefficients bn. 
# Part of that requires us to take some derivatives and find the 
# constants An and Bn for the homogenous functions.
#################################################################################

alist = np.zeros(motes)
blist = np.zeros(motes)

def compute_constants_ab():
    for i in range(motes):
        alist[i] = fourier_defl[i] + psi_particular(0, i)
        diff0 = diff(lambda z: psi_particular(z, i), 0, dx)
        blist[i] = (fourier_vel[i] - diff0) / alfas[i]

def time_coeff(t, n):
    alfa = alfas[n]
    y = alist[n]*cos(alfa*t) + blist[n]*sin(alfa*t) + psi_particular(t, n)
    return y

def plot_time(n):
    pts = 100
    tlist = np.linspace(0, 50, pts)
    ylist = np.zeros(pts)
    for i in range(pts):
        ylist[i] = time_coeff(tlist[i], n)
    plt.plot(tlist, ylist)

def plot_time_constants():
    xlist = range(motes)
    plt.plot(xlist, alist, label="A coeff")
    plt.plot(xlist, blist, label="B coeff")
    plt.xlabel("mote (n)")
    plt.legend()
    plt.show()

compute_constants_ab()
#plot_time_constants()
#plot_time(1)


#################################################################################
# Final solution.
#
# Having found all the components we turn our attention to plotting 
# the final deviation u(x,t).
#################################################################################

def deflection(x, t, n):
    y = 0
    for i in range(n):
        y += time_coeff(t, i) * phi_eigenfunc(x, i)
    return y

def plot_deflection_2d(t, n):
    pts = 100
    xlist = np.linspace(0, L, pts)

    ylist = p_map(lambda x: deflection(x,t,n), xlist, num_cpus=4)
    plt.plot(xlist, ylist)
    plt.show()

def plot_deflection_3d(tmax):
    x_pts = 10
    t_pts = 30

    tlist = np.linspace(0, tmax, t_pts)
    xlist = np.linspace(0, L, x_pts)
    X, T = np.meshgrid(xlist, tlist)

    Z = np.zeros((t_pts, x_pts))
    Z = np.array(p_map(
        lambda t: list(map(lambda x: deflection(x, t, motes),xlist)),
        tlist, num_cpus=4))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, T, Z, cmap="viridis")
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)');
    plt.show()

plot_deflection_3d(100)
