#!/bin/python
import scipy
import scipy.optimize as optimize
import scipy.integrate as integrate
from scipy import pi
import numpy as np
from numpy import cos, sin, cosh, sinh, exp, abs
import matplotlib.pyplot as plt
from p_tqdm import p_map
import json
import os


#################################################################################
# Settings.
#################################################################################

# Beam constants.
L = 200
E = 1 
I = 1
mu = 1
R = 1
ocean_density = 1030
inertia_coeff = 1
drag_coeff = 2

# Wave constants.
wave_period = 1
wave_amp = 1
wave_length = 1
depth = 100

# PDE constants.
motes = 10
quad_lowp = 20
dx = 1e-6
cpu_count = 4

def forcing(x, t):
    return cos(t/60.0*2*pi)

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
    return integrate.fixed_quad(lambda xl: list(map(f, xl)), a, b, n=quad_lowp)[0]

def integral_highp(f, a, b):
    return integrate.quad(f, a, b)[0]

# Tools for serializing numpy arrays and likes.
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(filename, obj):
    fh = open(filename, "w")
    json.dump(obj, fh, indent=1, cls=NumpyArrayEncoder)
    fh.close()

def load_json(filename):
    fh = open(filename, "r")
    data = json.load(fh)
    for k in data:
        if isinstance(data[k], list):
            data[k] = np.asarray(data[k])
    fh.close()
    return data


#################################################################################
# The wave equation.
#
# In this section we define the wave forcing function and 
# we analyse it with some plots.
#################################################################################

cross_section = pi * R**2
volume = cross_section * L
omega = 2*pi/wave_period
k = 2*pi/wave_length

def wave_vel(x, t):
#    C = exp(k*H)
#    y = sigma * wave_amp * (exp(k*z)*C + exp(-k*z)/C) / (C - 1/C) * cos(sigma*t)
    y = omega * wave_amp * sin(omega*t) * cosh(k*depth - k*x)/sinh(k*depth)
    return y

def wave_acc(x, t):
    y = omega**2 * wave_amp * cos(omega*t) * cosh(k*depth - k*x)/sinh(k*depth)
    return y

def morrison(x, t):
    v = wave_vel(x,t)
    inertia_f = ocean_density * inertia_coeff * volume * wave_acc(x,t)
    drag_f = 0.5 * drag_coeff * ocean_density * cross_section * v*abs(v)
    y = inertia_f + drag_f
    return y

def plot_wave_speed_2d(t):
    pts = 999
    zlist = np.linspace(0, depth/10, 999)
    ylist = np.zeros(pts)

    for i in range(pts):
        ylist[i] = wave_vel(zlist[i], t)

    plt.figure()
    plt.title("Wave speed at time t=%3.1f against depth" % t)
    plt.xlabel("depth (m)")
    plt.ylabel("speed (m/s)")
    plt.plot(zlist, ylist)

#plot_wave_speed_2d(0.5*pi/omega)
#plt.show()
#exit()


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
        c = pi * (i + 0.5)

        # Compute the root in normalized coordinates then scale back.
        yr = optimize.root_scalar(ev_function, bracket=[c-r, c+r],
                                  method="brentq").root
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
    plt.figure()
    for n in range(3):
        plot_eigenfunc(n)
    plt.title("First three motes of eigenfunctions.")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, L)

def plot_eigenfuncs_high():
    plt.figure()
    plot_eigenfunc(motes-1)
    plt.title("Highest mote (%d) of eigenfunctions." % (motes-1))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, L)

#plot_eigenfuncs()
#plot_eigenfuncs_high()
#plt.show()
#exit()


#################################################################################
# Inner product <phi, phi> numerically.
#
# We research here what happens if we integrate phi^2.
# This is needed for further computations.
#################################################################################

def phi_inner(n):
    return L

def phi_inner_num(n):
    return integral_highp(lambda x: phi_eigenfunc(x, n)**2, 0, L)

def test_phi_inner():
    for n in range(0, 10):
     print(phi_inner_num(n))

#test_phi_inner()
#exit()

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
    plt.figure()
    plt.plot(xlist, ylist)
    plt.plot(xlist, ylist2)
    plt.title("Eigenfunction expansion of function.")
    plt.xlabel("x")
    plt.ylabel("y")

#plot_fourier_series(lambda x: (x % 100.0) / 100.0)
#plt.show()
#exit()

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
    plt.plot(tlist, ylist, label="psi %d" % n)

def plot_psi_test(n):
    plt.figure()
    for i in range(n):
        plot_psi(i)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("First %d psi_n particular solutions." % n)

#plot_psi_test(5)
#plt.show()
#exit()


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
    plt.plot(tlist, ylist, label="b%d(t)" % n)

def plot_time_constants():
    xlist = range(motes)
    plt.figure()
    plt.plot(xlist, alist, label="A coeff")
    plt.plot(xlist, blist, label="B coeff")
    plt.title("Time constants An and Bn")
    plt.xlabel("n")
    plt.ylabel("y")
    plt.legend()

def plot_time_test(n):
    plt.figure()
    for i in range(n):
        plot_time(i)
    plt.title("First %d time solutions bn." % n)
    plt.xlabel("t")
    plt.ylabel("y")

compute_constants_ab()

#plot_time_constants()
#plot_time_test(5)
#plt.show()
#exit()


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

# Convenience methods.
def plot_deflection_2d(t, pts):
    xlist = np.linspace(0, L, pts)
    ylist = p_map(lambda x: deflection(x,t,motes), xlist, num_cpus=cpu_count)
    plt.figure()
    plt.title("Deflection u(x,t) at time t=%3.1f in space." % t)
    plt.xlabel("x (meters)")
    plt.ylabel("u (meters)")
    plt.plot(xlist, ylist)

def plot_deflection_point_2d(x, tstart, tend, pts):
    tlist = np.linspace(tstart, tend, pts)
    ylist = p_map(lambda t: deflection(x,t,motes), tlist, num_cpus=cpu_count)
    plt.figure()
    plt.title("Deflection u(x,t) at x=L in time.")
    plt.xlabel("t (seconds)")
    plt.ylabel("u (meters)")
    plt.plot(tlist, ylist)

def compute_deflection_3d(tstart, tend, x_pts, t_pts):
    tlist = np.linspace(tstart, tend, t_pts)
    xlist = np.linspace(0, L, x_pts)
    X, T = np.meshgrid(xlist, tlist)

    Z = np.zeros((t_pts, x_pts))
    Z = np.array(p_map(
        lambda t: list(map(lambda x: deflection(x, t, motes),xlist)),
        tlist, num_cpus=cpu_count))
    return { "X": xlist, "T": tlist, "Z": Z }

def plot_deflection_3d_data(data):
    X, T = np.meshgrid(data["X"], data["T"])
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, T, data["Z"], cmap="viridis", rstride=1, cstride=1)
    ax.set_title("Deflection u(x,t) in space and time.")
    ax.set_xlabel('x (meters)')
    ax.set_ylabel('t (seconds)')
    ax.set_zlabel('u (meters)');

def plot_deflection_3d(tstart, tend, x_pts, t_pts):
    data = compute_deflection_3d(tstart, tend, x_pts, t_pts)
    plot_deflection_3d_data(data)

def plot_deflection_heatmap(data, interp="spline36"):
    plt.figure()
    plt.title("Heatmap of deflection u(x,t) in space and time")
    plt.xlabel("t (seconds)")
    plt.ylabel("x (meters)")
    z_trans = np.transpose(data["Z"])
    extents = (min(data["T"]), max(data["T"]), 0, L)
    plt.imshow(z_trans,
               aspect="auto",
               origin="lower",
               extent=extents,
               interpolation=interp)
    plt.colorbar(label="Deflection (meters)")


if __name__ == "__main__":
    # ***Delftblue compute jobs***
    # Only run this on delftblue since your computer will go brr.
    #save_json("data/3d_hires.json", compute_deflection_3d(0, 600, 50, 200))

    # High precision single sample test.
    #plot_deflection_2d(440, 200)
    #plot_deflection_3d(8, 14, 100, 10)

    # Periodicity test.
    #plot_deflection_point_2d(L, 0, 300, 100)

    # Overview plot.
    #plot_deflection_3d(0, 300, 10, 50)
    #plot_deflection_3d_data(load_json("delftblue_data/3d_hires.json"))

    # ** Heatmaps **
    # Be careful with heatmaps that the interpolation mode (default "spline36")
    # is not giving false impressions of the data. If not certain, use "nearest". 
    # Then the heatmap becomes pixellated but the data is presented as-is.
    #plot_deflection_heatmap(load_json("data/3d_test.json"), "nearest")
    #plot_deflection_heatmap(load_json("data/3d_test.json"), "spline36")
    plot_deflection_heatmap(load_json("delftblue_data/3d_hires.json"))

    # Plot the results.
    plt.show()
