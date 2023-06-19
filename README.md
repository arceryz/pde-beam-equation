# Euler-Bernoulli beam equation python.

![resonance](/pictures/c3_def.gif)

This github repository contains the necessary scripts for simulating a wind
turbine in the ocean subject to forces of waves (Morison equations) and forces
from earthquakes.

The script can compute datasets for various scenarios and then plot them in an
visualiser script called "Windmolentje". This visualiser shows the beam in real
time being deflected, the excitation of various motes and plots of forces.

There is only a few dependencies:
- matplotlib
- numpy
- scipy
- p_tqdm from [https://github.com/swansonk14/p_tqdm](https://github.com/swansonk14/p_tqdm).
  which can be installed with `pip install p_tqdm`. This package is used for 
  parallel computing with progress bar.

The simulations are computed in parallel with as many threads as specified by
`cpu_count` default to 4. If running on a stronger computer make sure to
increase this parameter. Modify `beam.py` if you want to change the scenario and
settings of the simulations. All plotting tools are inside `plots.py`.

More resources can be found on
[wikipedia](https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory).
The mathematics underlying these scripts/formulas are all derived in a neat
mathematical report (not yet available). This contains all the derivations for
formulas implemented numerically here. Before programming there was lots of
analytical solving, it is not magic.
