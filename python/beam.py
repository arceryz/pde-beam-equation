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
# Constants.
#################################################################################

# Beam settings.
L = 150
H = 50
E = 200*1e+9
R1 = 2.2
R2 = 2.25
rho_steel = 5487

# Morisson settings.
Ca = 0.33496
Cd = 1.17

# Wave settings.
rho_sea = 1030
wave_period = 5.7
wave_amp = 1.5
wave_length = 33.8

# Numerical settings.
motes = 5
quad_lowp = 20
dx = 1e-6
cpu_count = 4


#################################################################################
# Dependent variables.
#################################################################################

I = pi / 4 * (R2**4 - R1**4)
sigma = 2*pi/wave_period
k = 2*pi/wave_length
V = pi*(R2**2 - R1**2)
mu = rho_steel * V

S = sinh(k*H)
A_wave = sigma * wave_amp
A_iner = rho_sea * (1+ Ca) * V
A_drag = 0.5 * rho_sea * Cd * V
B_iner = -A_iner * A_wave * sigma
B_drag = A_drag * A_wave**2


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
# Initital conditions and forcing.
#
# In this section we define the wave forcing function and 
# we analyse it with some plots.
#################################################################################

def forcing(x, t):
    return morisson(x, t)

def ic_deflection(x):
    return 0

def ic_velocity(x):
    return 0


#################################################################################
# The wave equation.
#
# In this section we define the wave forcing function and 
# we analyse it with some plots.
#################################################################################

def wave_vel(x, t):
    y = A_wave * cos(sigma*t) * cosh(k*x)/S
    return y

def wave_acc(x, t):
    y = -A_wave*sigma * sin(sigma*t) * cosh(k*x)/S
    return y

def morisson(x, t):
    if x > H:
        return 0
#    v = wave_vel(x,t)
#    y = A_iner*wave_acc(x,t) + A_drag *v*abs(v)
    y = B_iner * sin(sigma*t)*cosh(k*x)/S + \
        B_drag*cos(sigma*t)*abs(cos(sigma*t))*(cosh(k*x)/S)**2
    return y

def plot_wave_speed_2d(t):
    pts = 999
    zlist = np.linspace(0, H, 999)
    ylist = np.zeros(pts)

    for i in range(pts):
        ylist[i] = wave_vel(zlist[i], t)

    plt.figure()
    plt.title("Wave speed at time t=%3.1f against H" % t)
    plt.xlabel("height from ocean floor (m)")
    plt.ylabel("speed (m/s)")
    plt.plot(zlist, ylist)

def plot_morisson_2d(t):
    pts = 999
    zlist = np.linspace(0, L, 999)
    ylist = np.zeros(pts)

    for i in range(pts):
        ylist[i] = morisson(zlist[i], t)
    plt.figure()
    plt.plot(zlist, ylist, label="t=%3.1f" % t)

def plot_morisson_3d(tstart, tend, x_pts, t_pts):
    tlist = np.linspace(tstart, tend, t_pts)
    xlist = np.linspace(0, L, x_pts)
    X, T = np.meshgrid(xlist, tlist)

    Z = np.zeros((t_pts, x_pts))
    Z = np.array(p_map(
        lambda t: list(map(lambda x: morisson(x, t),xlist)),
        tlist, num_cpus=cpu_count))
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, T, Z, cmap="viridis", rstride=1, cstride=1)
    ax.set_title("Morrison m(x,t) in space and time.")
    ax.set_xlabel('x (meters)')
    ax.set_ylabel('t (seconds)')
    ax.set_zlabel('force (newton)');


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


#################################################################################
# Computation of morisson fourier fractions.
#
#################################################################################

mf_iner = np.zeros(motes)
mf_drag = np.zeros(motes)

def morisson_fract_iner(n):
    y = integral_highp(lambda x: cosh(k*x)/S*phi_eigenfunc(x,n), 0, H)
    return y

def morisson_fract_drag(n):
    y = integral_highp(lambda x: (cosh(k*x)/S)**2*phi_eigenfunc(x,n), 0, H)
    return y

def compute_mfracts():
    for i in range(motes):
        mf_iner[i] = morisson_fract_iner(i)
        mf_drag[i] = morisson_fract_drag(i)
    pass


#################################################################################
# Fourier coefficients computation.
#
# In this section we compute the integrals for fourier coefficients of
# arbitrary functions. We test this with some sample functions.  
# Then we store the coefficients we need in lists for later use.
#################################################################################

def fourier_coeff(f, n) -> float:
    y = integral_lowp(lambda x: f(x)*phi_eigenfunc(x, n), 0, H) / L
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

fourier_defl = []
fourier_vel  = []


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

    step = 0.5
    points = np.append(np.arange(0, t, step), [ t ])
    y = 0

    f = lambda s: (B_iner*mf_iner[n]*sin(sigma*t) + \
                   B_drag*mf_drag[n]*cos(sigma*t) * abs(cos(sigma*t)))*sin(alfa*(t-s))
    for i in range(len(points)-1):
        a = points[i]
        b = points[i+1]
        y += 1/alfa*integral_highp(f, points[i], points[i+1])
    return y

def plot_psi(n):
    pts = 500
    tlist = np.linspace(0, 30, pts)
    ylist = p_map(lambda t: psi_particular(t,n), tlist)
    plt.plot(tlist, ylist, label="psi %d" % n)

def plot_psi_test(n):
    plt.figure()
    for i in range(n):
        plot_psi(i)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("First %d psi_n particular solutions." % n)


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


#################################################################################
# Final solution.
#
# Having found all the components we turn our attention to plotting 
# the final deviation u(x,t).
#################################################################################

def deflection(x, t):
    tvec = compute_time_vec(t)
    xvec = compute_space_vec(x)
    return np.inner(tvec, xvec)

def compute_time_vec(t):
    v = np.zeros(motes)
    for i in range(motes):
        v[i] = time_coeff(t, i)
    return v

def compute_space_vec(x):
    v = np.zeros(motes)
    for i in range(motes):
        v[i] = phi_eigenfunc(x, i)
    return v

def plot_deflection_point_2d(x, tstart, tend, pts):
    xvec = compute_space_vec(x)
    tlist = np.linspace(tstart, tend, pts)
    ylist = p_map(lambda t: np.inner(compute_time_vec(t), xvec), tlist, num_cpus=cpu_count)
    plt.figure()
    plt.ylim(-15, 15)
    plt.title("Deflection u(x,t) at x=L in time.")
    plt.xlabel("t (seconds)")
    plt.ylabel("u (meters)")
    plt.plot(tlist, ylist)

def compute_deflection_3d(tstart, tend, x_pts, t_pts):
    tlist = np.linspace(tstart, tend, t_pts)
    xlist = np.linspace(0, L, x_pts)

    xvecs = list(map(lambda x: compute_space_vec(x), xlist))
    tvecs = list(p_map(lambda t: compute_time_vec(t), tlist,num_cpus=cpu_count))

    X, T = np.meshgrid(xlist, tlist)

    Z = np.zeros((t_pts, x_pts))
    Z = np.array(p_map(
        lambda ti: list(map(lambda xi: np.inner(xvecs[xi], tvecs[ti]), range(x_pts))),
        range(t_pts), num_cpus=cpu_count))
    return { "X": xlist, "T": tlist, "Z": Z }

def plot_deflection_3d_data(data):
    X, T = np.meshgrid(data["X"], data["T"])
    plt.figure()
    ax = plt.axes(projection="3d")
    bd = 1
    ax.set_zlim(-bd, bd)
    ax.plot_surface(X, T, data["Z"], cmap="viridis", rstride=1, cstride=1,
                    vmin=-bd, vmax=bd)
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


def compute_all():
    compute_evs()
    global fourier_defl
    global fourier_vel
    fourier_defl = fourier_coeffs_arr(ic_deflection, motes)
    fourier_vel  = fourier_coeffs_arr(ic_velocity, motes)
    compute_mfracts()
    compute_constants_ab()
