#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

def f(x, t):
    return np.sin(x + t)

def plot_func(t, f, plot, xlist):
    # Create a new plot data.
    yl = np.zeros(len(xlist))
    for i in range(len(xlist)):
        yl[i] = f(xlist[i], t)
    # Set the new data and allow it to render.
    plot.set_data(yl, xlist)
    return [ plot ]

def do_plot():
    fig = plt.figure()
    ax = plt.axes()

    xlist = np.linspace(0, 2*np.pi, 100)
    ax.set_ylim(0, 2*np.pi)
    ax.set_xlim(-1, 1)

    # Create a plot.
    plot = ax.plot([], [])[0]

    numframes = 300
    tmax = 10
    tlist = np.linspace(0, tmax, numframes)
    frametime = int(float(tmax) / numframes * 1000)
    print(frametime)

    ufunc = partial(plot_func, f=f, plot=plot, xlist=xlist)
    ani = FuncAnimation(fig, ufunc, frames=tlist, blit=True, interval=frametime)
    plt.show()

do_plot()
