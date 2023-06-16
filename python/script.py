#!/bin/python
from beam import *

compute_all()

if __name__ == "__main__":
    __spec__ = None

    # Morison test.
    #print("Mass = %.2f * %.2f = %.2f kg" % (L,mu, L*mu))
    #plot_wave_time(20)
    #plot_morisson_2d_time(20)

    # Periodicity test.
    #print(eigenvalues*L)
    #plot_deflection_point_2d(L, 0, 10, 50)

    # Overview plot.
    #data = compute_deflection_3d(0, 60, 100, 100)
    #save_json("data/backup.json" ,data)
    data = load_json("data/resonance_period_5_60.json")
    plot_deflection_3d_data(data)
    #plot_deflection_3d_data(load_json("delftblue_data/3d_hires.json"))

    # ** Heatmaps **
    # Be careful with heatmaps that the interpolation mode (default "spline36")
    # is not giving false impressions of the data. If not certain, use "nearest". 
    # Then the heatmap becomes pixellated but the data is presented as-is.
    #plot_deflection_heatmap(load_json("data/3d_test.json"), "nearest")
    #plot_deflection_heatmap(load_json("data/3d_test.json"), "spline36")
    #plot_deflection_heatmap(load_json("delftblue_data/3d_hires.json"))
    plot_deflection_heatmap(data)

    # ***Delftblue compute jobs***
    # Only run this on delftblue since your computer will go brr.
    #save_json("data/3d_hires.json", compute_deflection_3d(0, 600, 50, 200))
    plt.show()
