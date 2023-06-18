#!/bin/python
from plots import *

compute_evs()
compute_betas()

if __name__ == "__main__":
    __spec__ = None
    # Morison test.
    #print("Mass = %.2f * %.2f = %.2f kg" % (L,mu, L*mu))
    #plot_wave_time(20)
    #plot_morisson_2d_time(20)

    # Periodicity test.
    #print(eigenvalues*L)
    #plot_deflection(L, 0, 30, 1000)
    #plot_time_test(1, 30, 999)

    # Animations.
    #data = load_json("data/resonance_period_5_60.json")
    data = compute_deflection_3d(0, 60, 100, 500)
    ani = anim_deflection(data, 10)

    # Overview plot.
    #data = compute_deflection_3d(0, 60, 100, 100)
    #save_json("data/backup.json" ,data)
    #plot_deflection_3d(data)
    #plot_deflection_3d(load_json("delftblue_data/3d_hires.json"))

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
