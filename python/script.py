#!/bin/python
from plots import *

compute_evs()
compute_betas()

if __name__ == "__main__":
    __spec__ = None
    print("* Windmolentje *")
    print("Alfas: %s" % str(alfas / (2*pi)))

    # Animations.
    #data = compute_deflection_3d(0, 60, 100, 999)
    #save_json("data/optim/storm_60_hr.json", data)

    data2 = load_json("data/optim/storm_60_hr.json")
    anim_deflection(data2, 1)
