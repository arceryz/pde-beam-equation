import json
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
from matplotlib.animation import FuncAnimation
from beam import *
from functools import partial


#
# JSON utility.
#
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



#
# Morison plots.
#
def plot_wave_time(tend):
	pts = 999
	tlist = np.linspace(0, tend, 999)
	ylist = np.zeros(pts)
	ylist2 = np.zeros(pts)

	for i in range(pts):
		ylist[i] = wave_vel(H, tlist[i])
		ylist2[i] = wave_acc(H, tlist[i])

	plt.figure()
	plt.title("Waves at x=H (surface) in time.")
	plt.xlabel("height from ocean floor (m)")
	plt.ylabel("speed (m/s)")
	plt.plot(tlist, ylist, label="Speed")
	plt.plot(tlist, ylist2, label="Acceleration")
	plt.legend()

def plot_wave_speed_2d(t):
	pts = 999
	zlist = np.linspace(0, H, 999)
	ylist = np.zeros(pts)

	for i in range(pts):
		ylist[i] = wave_vel(zlist[i], t)

	plt.figure()
	plt.title("Wave speed at time t=%3.1f against z" % t)
	plt.xlabel("height from ocean floor (m)")
	plt.ylabel("speed (m/s)")
	plt.plot(zlist, ylist)

def plot_morison_2d_time(tend):
	pts = 999
	tlist = np.linspace(0, tend, 999)
	ylist = np.zeros(pts)
	ylist_iner = np.zeros(pts)
	ylist_drag = np.zeros(pts)

	for i in range(pts):
		ylist[i] = morison(H, tlist[i])
		ylist_iner[i] = f_iner(H, tlist[i])
		ylist_drag[i] = f_drag(H, tlist[i])
	plt.figure()

	A = rho_sea * (1+ Ca) * pi/4 * D**2
	B = 0.5 * rho_sea * Cd * D
	plt.title("Morison forces Inertia=%.2f, Drag=%.2f" % (A, B))
	plt.plot(tlist, ylist, label="Morison Force")
	plt.plot(tlist, ylist_iner, label="Inertia Force")
	plt.plot(tlist, ylist_drag, label = "Drag Force")
	plt.ylabel("Force (N)")
	plt.xlabel("Time t")
	plt.legend()

def plot_morison_2d(t):
	pts = 999
	zlist = np.linspace(0, L, 999)
	ylist = np.zeros(pts)

	for i in range(pts):
		ylist[i] = morison(zlist[i], t)
	plt.figure()
	plt.plot(zlist, ylist, label="t=%3.1f" % t)

def plot_morison_3d(tstart, tend, x_pts, t_pts):
	tlist = np.linspace(tstart, tend, t_pts)
	xlist = np.linspace(0, H, x_pts)
	X, T = np.meshgrid(xlist, tlist)

	Z = np.zeros((t_pts, x_pts))
	Z = np.array(p_map(
		lambda t: list(map(lambda x: morison(x, t),xlist)),
		tlist, num_cpus=cpu_count))
	plt.figure()
	ax = plt.axes(projection="3d")
	ax.plot_surface(X, T, Z, cmap="viridis", rstride=1, cstride=1)
	ax.set_title("Morrison m(x,t) in space and time.")
	ax.set_xlabel('x (meters)')
	ax.set_ylabel('t (seconds)')
	ax.set_zlabel('force (newton)');



#
# Eigenfunctions.
#
def plot_eigenfunc(n):
	pts = 999
	xlist = np.linspace(0, L, pts)
	ylist = np.zeros(pts)
	for i in range(pts):
		ylist[i] = phi_eigenfunc(xlist[i], n)
	plt.plot(xlist, ylist)

def plot_eigenfuncs(n):
	plt.figure()
	for i in range(n):
		plot_eigenfunc(i)
	plt.title("First %d motes of eigenfunctions." % n)
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



#
# Time coefficients/psi.
#
def plot_time_parts(t, n):
	pts = 100
	alfa = alfas[n]
	slist = np.linspace(0, t, pts)
	ylist = p_map(lambda s: fourier_force(s,n), slist)
	y2list = p_map(lambda s: sin(alfa*(t-s)), slist)
	y3list = np.zeros(pts)
	for i in range(pts):
		y3list[i] = ylist[i]*y2list[i]/alfa

	f = lambda s: fourier_force(s, n) * sin(alfa*(t-s))
	y4list = p_map(lambda s: 1/alfa*integral_highp(f, 0, s), slist)

	plt.xlabel("s")
	plt.ylabel("y")
	plt.title("Time parts for t=%3.2f, alfa=%3.2f" % (t, alfa)) 
	plt.plot(slist, ylist, label="Fourier force")
	plt.plot(slist, y2list, label="Sin(alfa(t-s))")
	plt.plot(slist, y3list, label="Integrand/alfa")
	plt.plot(slist, y4list, label="Integral until s")
	plt.axhline(y=0.0, color='b')
	plt.legend()

def plot_time(n, tend, pts):
	tlist = np.linspace(0, tend, pts)
	ylist = p_map(lambda t: time_coeff(t,n), tlist)

	maxy = max(ylist)
	maxindex = ylist.index(maxy)
	maxt = tlist[maxindex]
	print("Time %d max value y=%.2f at t=%.2f" % (n, maxy, maxt))
	plt.plot(tlist, ylist, label="psi %d" % n)

def plot_time_test(n, tend, pts):
	plt.figure()
	for i in range(n):
		plot_time(i, tend, pts)
	plt.legend()
	plt.xlabel("t")
	plt.ylabel("y")
	plt.title("First %d b_n particular solutions." % n)




#
# Deflection
#
def plot_deflection(x, tstart, tend, pts):
    xlist = [ x ]
    tlist = np.linspace(tstart, tend, pts)
    ylist = np.transpose(deflection(xlist, tlist)[0])[0]

    print("Max deflection=%f" % max(ylist))

    plt.figure()
	#plt.ylim(-defl_norm, defl_norm)
    plt.title("Deflection u(x,t) at x=L in time.")
    plt.xlabel("t (seconds)")
    plt.ylabel("u (meters)")
    plt.plot(tlist, ylist)
    plt.show()

def compute_deflection_3d(tstart, tend, x_pts, t_pts):
    tlist = np.linspace(tstart, tend, t_pts)
    xlist = np.linspace(0, L, x_pts)
    X, T = np.meshgrid(xlist, tlist)
    Z, tvecs, xvecs, ae_diff, mor = deflection(xlist, tlist) 

    out = {
        "X": xlist,
        "T": tlist,
        "Z": Z,
        "tvecs": tvecs,
        "xvecs": xvecs,
        "f_ae": ae_diff,
        "f_mor": mor
    }
    return out

def plot_deflection_3d(data):
	X, T = np.meshgrid(data["X"], data["T"])
	plt.figure()
	ax = plt.axes(projection="3d")
	ax.set_zlim(-defl_norm, defl_norm)
	ax.plot_surface(X, T, data["Z"], cmap="viridis", rstride=1, cstride=1,
				 vmin=-defl_norm, vmax=defl_norm)
	ax.set_title("Deflection u(x,t) in space and time.")
	ax.set_xlabel('x (meters)')
	ax.set_ylabel('t (seconds)')
	ax.set_zlabel('u (meters)');

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
			vmin=-defl_norm, vmax=defl_norm,
			extent=extents,
			interpolation=interp)
	plt.colorbar(label="Deflection (meters)")



#
# Animated deflection plots.
#
def update_defl_frame(i, plots, axes, data):
    tlist = data["T"]
    xlist = data["X"]
    zlist = data["Z"][i]
    tv = data["tvecs"][i]
    umax = zlist[-1]

    plots[4].set_text("t=%3.2f, u(L,t)=%3.2f" % (tlist[i], umax))
    plots[3].set_xdata(tlist[i])
    plots[0].set_data(zlist, xlist)
    plots[5].set_xdata(tlist[i])
    plots[6].set_xdata(tlist[i])
    return plots

def anim_deflection(data, speed=1):

    tlist = data["T"]
    tstart = min(tlist)
    tend = max(tlist)
    numframes = len(tlist)

    # Compute the data needed to plot animation.
    margin = 20
    fig, axes = plt.subplots(2, 2)

    umin = data["Z"].min()
    umax = data["Z"].max()
    fig.suptitle("Windmolentje. Scenario=\"%s\". Extremes %3.2fm < u < %3.2fm" %
                 (scenario, umin, umax))
    im = image.imread("windmolentje.png")
    newax = fig.add_axes([0.0,0.9,0.1,0.1], anchor='NW', zorder=1)
    newax.imshow(im)
    newax.axis('off')

    ax = axes[0][0]
    ax.set_title("Simulation of windturbine at sea")
    ax.set_xlim(-0.5*L, 0.5*L)
    ax.set_ylim(0, L+margin)
    ax.set_xlabel("deflection (m)")
    ax.set_ylabel("height (m)")
    ax.set_aspect("equal")

    # Create the plot object.
    plot = ax.plot([], [], lw=72.0*R2/10, color="gray", label="Windturbine")[0]
    frametime = float(tend - tstart) / numframes * 1000 / speed
    print(frametime)

    ax2 = axes[0][1]
    ax2.set_title("Deflection spectrum")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("height (m)")
    ax2.set_aspect("equal")

    ax3= axes[1][0]
    ax3.set_title("Excitation of eigenfunction motes")

    for k in range(motes):
        ylist = np.zeros(len(tlist))
        for i in range(len(tlist)):
            tvecs = data["tvecs"][i]
            ylist[i] = tvecs[k]
        ax3.plot(tlist, ylist, label="Mote %d" % (k+1))
    ax3.legend(loc="upper right")

    # Force plot.
    ax4 = axes[1][1]
    ax4.plot(tlist, data["f_mor"], label="Morison force at x=H")
    ax4.plot(tlist, data["f_ae"], label="Earthquake force per meter")
    ax4.set_title("Morison and Earthquake forces in time")
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("Force (N)")
    ax4.legend(loc="upper right")

    plots = [ 
        plot, 
        ax.axhline(H, color="blue", ls="-", label="Water surface"), 
        ax.axvline(0, color="red", ls="-", label="Center"),
        ax2.axvline(0, color="red"),
        ax.text(-0.5*L+5, L+margin-5, "Hello", ha="left", va="center",
                color="black", fontsize=10),
        ax3.axvline(0, color="red"),
        ax4.axvline(0, color="red"),
    ]
    ufunc = partial(update_defl_frame, plots=plots, axes=axes, data=data)
    ani = FuncAnimation(fig, ufunc, frames=range(numframes), blit=True, interval=frametime)

    z_trans = np.transpose(data["Z"])
    extents = (min(data["T"]), max(data["T"]), 0, L)
    im = ax2.imshow(z_trans,
            aspect="auto",
            origin="lower",
            vmin=-defl_norm, vmax=defl_norm,
            extent=extents,
            interpolation="spline36")
    plt.colorbar(im, label="Deflection (meters)", ax=ax2)
    plt.show()
