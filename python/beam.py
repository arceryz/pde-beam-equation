import scipy
import scipy.optimize as optimize
import scipy.integrate as integrate
import numpy as np
import os
from numpy import cos, sin, cosh, sinh, exp, abs, pi
from p_tqdm import p_map


#
# Beam settings.
#
L = 150
H = 50
E = 210*1e+9
R1 = 2.2
R2 = 2.25
rho_steel = 7850



#
# Morison settings.
#
morison_enabled = True
Ca = 0.33496
Cd = 1.1
rho_sea = 1030
scenario = "rough"



#
# Sea scenarios. Put here the various types of seas we want to test in our
# model. The syntax is "sea_name": (period, amplitude, wavelength). 
# All units in seconds or meters.
#
sea_scenarios = {
    "mild" : (5.7, 1.5, 33.8),
    "medium":(8.6,4.1, 76.5),
    "rough": (11.4,8.5,136)
}



#
# Earthquake settings. 
# The earthquakes are provided in a list specifying (Amplitude, Hz).
# Multiple earthquakes can be computed for no additional cost!
#
earthquakes = [
   (1, 0.10), (0.5,1),(0.1,10), #II  
   (2,0.10), (1,1), (0.2,10)#III
]


#
# Constant forcing. This the constant external forcing term.
# For testing with the analytical model.
#
F_constant = 0


#
# Numerical settings.
# Setting then number of motes higher is expensive.
#
motes = 5
quad_lowp = 20
dx = 1e-6
cpu_count = 4
defl_norm = 10



#
# These variables dependent on the above and should not be modified.
#
wave_period, wave_amp, wave_length = sea_scenarios[scenario]
I = (pi / 4 * (R2**4 - R1**4))
sigma = 2*pi/wave_period
k = 2*pi/wave_length
Volume = pi*R2**2
Area = pi*(R2**2 - R1**2)
D = 2*R2
mu = rho_steel * Area



#
# The following are some utility methods.
# Integration and differentiation.
#
def diff(f, x):
    return (f(x) + f(x+dx))/dx

def integral(f, a, b):
    return integrate.quad(f, a, b, limit=100)[0]




#
# The wave equation.
#
# In this section we define the wave forcing function and 
# we analyse it with some plots.
#
def wave_vel(x, t):
    amp = sigma * wave_amp * cosh(k*x)/sinh(k*H)
    y = amp * cos(sigma*t)
    return y

def wave_acc(x, t):
    amp = sigma * wave_amp * cosh(k*x)/sinh(k*H)
    y = -sigma * amp * sin(sigma*t)
    return y

def f_iner(x, t):
    return rho_sea * (1+ Ca) * pi/4 * D**2 * wave_acc(x,t)

def f_drag(x, t):
    v = wave_vel(x,t)
    return 0.5 * rho_sea * Cd * D *v*abs(v)

def morison(x, t):
    if x > H:
        return 0
    return f_iner(x,t) + f_drag(x, t)




#
# The earthquake.
# This is the summed earthquake influence.
# The derivatives of the earthquake are already inserted in the 
# equations and are all optimised.
# Do not change this without changing the model!
#
def a_earthquake(t):
    y = 0
    for E_amp, E_freq in earthquakes:
        y += E_amp * cos(E_freq * 2*pi * t)
    return y

def a_earthquake_diff(t):
    y = 0
    for E_amp, E_freq in earthquakes:
        w = E_freq * 2*pi
        y += -mu * E_amp * w**2 * cos(w * t)
    return y


#
# Computing the eigenvalues.
#
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
        alfas[i] = xr**2 * np.sqrt(E*I / mu)



#
# The eigenfunctions and betas.
#
def phi_eigenfunc(x, n):
    ev = eigenvalues[n]
    c = (cosh(ev*L) + cos(ev*L)) / \
        (sinh(ev*L) + sin(ev*L))
    y = ((1-c)*exp(ev*x) + (1+c)*exp(-ev*x))/2.0 + c*sin(ev*x) - cos(ev*x)
    return y

betas = np.zeros(motes)
betas_cosh1 = np.zeros(motes)
betas_cosh2 = np.zeros(motes)

def compute_betas():
    print("Computing betas")
    for n in range(motes):
        betas[n] = integral(lambda x: phi_eigenfunc(x, n), 0, L) / L
        betas_cosh1[n] = integral(lambda x: cosh(k*x) * phi_eigenfunc(x, n), 0, H) / L
        betas_cosh2[n] = integral(lambda x: cosh(k*x)**2 * phi_eigenfunc(x, n), 0, H) / L
    print(betas_cosh1)
    print(betas_cosh2)


#
# The time coefficients.
# We neglect the initial conditions as zero to simplify bn.
#
def morison_eigencoeff(t, n):
    sk = sinh(k*H)

    C_iner = pi * R2 ** 2 * rho_sea * (1+Ca) * -sigma**2/sk*wave_amp * betas_cosh1[n]
    fi = C_iner * sin(sigma*t)

    C_drag = R2*rho_sea*Cd * (sigma**2)*(wave_amp**2)/(sk**2) * betas_cosh2[n]
    fd = C_drag * cos(sigma*t) * abs(cos(sigma*t))

    return fd + fi

def resonance_morison(t, n):
    y_mor = 0
    alfa = alfas[n]
    if morison_enabled:
        f = lambda s: morison_eigencoeff(s, n) * sin(alfa*(t-s))
        y_mor = integral(f, 0, t)
    return y_mor


#
# The resonance equation for sines is the integral <cos, sin> over t.
# For cosines it reduces to this expression.
# We call this integral in general the "Resonance" integral because it 
# determines the resonance.
#
def resonance_cos(w, a, t):
    return a * (cos(w*t) - cos(a*t)) / (a**2 - w**2)

def time_coeff(t, n):
    alfa = alfas[n]
    beta = betas[n]


    # The morison resonance contribution.
    y_mor = resonance_morison(t, n)

    # The earthquake resonance contribution.
    # If there are no earthquakes then y_eq remains zero and 
    # we are just left with 1/(alfa*mu) * y_mor.
    # This is the same as the previous case basically.
    y_eq = 0
    for E_amp, E_freq in earthquakes:
        freq = E_freq * 2*pi
        y_eq += E_amp * freq**2 * resonance_cos(freq, alfa, t)
    y_eq *= mu * beta

    y_const = beta * F_constant * resonance_cos(0, alfa, t)

    y = 1/(alfa*mu) * (y_mor + y_const + y_eq)
    return y


#
# Deflection computation. 
# Here we include the methods to compute the final deflection in 
# optimized way.
#
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

def deflection(xrange, trange):
    # Compute all spatial vectors and time vectors in the domain.
    print("Computing xvecs tvecs")
    xvecs = list(map(lambda x: compute_space_vec(x), xrange))
    tvecs = list(p_map(lambda t: compute_time_vec(t), trange, 
                       num_cpus=cpu_count))

    # Compute all earthquake influences.
    print("Computing earthquakes")
    ae_list = list(map(lambda t: a_earthquake(t), trange))
    print("Computing earthquakes forces")
    ae_list_diff = list(map(lambda t: a_earthquake_diff(t), trange))
    print("Computing morison forces")
    morison_list = list(map(lambda t: morison(H, t), trange))

    # Now compute the real u to get the result.
    print("Computing inner products")
    Z = np.array(list(map(
        lambda ti: list(map(
            lambda xi: np.inner(xvecs[xi], tvecs[ti]), 
            range(len(xrange)))),
        range(len(trange)))))
    return Z, tvecs, xvecs, ae_list_diff, morison_list
