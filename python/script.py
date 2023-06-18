#!/bin/python
from plots import *

compute_evs()
compute_betas()

if __name__ == "__main__":
    __spec__ = None
    print("* Windmolentje *")
    print("Alfas: %s" % str(alfas))

    # Animations.
    data = load_json("data/rough_with_eq1.json")
    #data = compute_deflection_3d(0, 30, 100, 100)
    anim_deflection(data, 10)

    #save_json("data/backup.json", data)
