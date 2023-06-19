#!/bin/python
from plots import *

compute_evs()
compute_betas()

if __name__ == "__main__":
    __spec__ = None
    print("* Windmolentje *")
    print("Alfas: %s" % str(alfas))

    # Animations.
    #data = load_json("data/VERANDEREN.json")
    data = compute_deflection_3d(0, 60, 100, 120)
    save_json("data/ExtensionShae/AII_EI_Default.json", data)
    
    anim_deflection(data, 10)