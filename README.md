# Windturbines at sea subject to waves and earthquakes.

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

The mathematics underlying this script is all described in detail in our report
that you can find in this repository at [/Report.pdf](/Report.pdf). 
More resources can be found on
[wikipedia](https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory).
