#!/bin/python
from beam import *

compute_all()

if __name__ == "__main__":
    __spec__ = None

    #plot_eigenfuncs(motes)
    #plot_morisson_3d(0, 30, 100, 100)
    #plot_time_constants()
    plot_psi_parts(8.74, 0)
#    print(psi_particular(8.74, 1))
    #plot_psi_test(1)
    plt.show()
    #exit()
    
    # Periodicity test.
    #plot_deflection_point_2d(L, 0, 30, 100)

    # Overview plot.
    #data = compute_deflection_3d(0, 30, 100, 100)
    #save_json("data/backup.json" ,data)
    #plot_deflection_3d_data(data)
    #plot_deflection_3d_data(load_json("delftblue_data/3d_hires.json"))

    # ** Heatmaps **
    # Be careful with heatmaps that the interpolation mode (default "spline36")
    # is not giving false impressions of the data. If not certain, use "nearest". 
    # Then the heatmap becomes pixellated but the data is presented as-is.
    #plot_deflection_heatmap(load_json("data/3d_test.json"), "nearest")
    #plot_deflection_heatmap(load_json("data/3d_test.json"), "spline36")
    #plot_deflection_heatmap(load_json("delftblue_data/3d_hires.json"))
    #plot_deflection_heatmap(data)

    # ***Delftblue compute jobs***
    # Only run this on delftblue since your computer will go brr.
    #save_json("data/3d_hires.json", compute_deflection_3d(0, 600, 50, 200))
    plt.show()
