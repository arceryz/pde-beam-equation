#!/bin/python
from plots import *

compute_evs()
compute_betas()

if __name__ == "__main__":
    __spec__ = None
    print("* Windmolentje *")
    print("Alfas: %s" % str(alfas))

    # Animations.
    #data = load_json("data/resonance_period_5_60.json")
    data = compute_deflection_3d(0, 60, 100, 500)
    ani = anim_deflection(data, 10)
    plt.show()
