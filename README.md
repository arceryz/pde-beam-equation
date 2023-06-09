# Euler-Bernoulli beam equation python.

![resonance](/pictures/resonance.png)

This github repository contains the necessary scripts for simulating the
Euler-Bernoulli beam equation with nonhomogenous forcing and fixed-free boundary
conditions. The scripts are dedicated to analysing the eigenfrequencies of a
windmill beam subject to force from the ocean.

There is only a few dependencies:
- matplotlib
- numpy
- scipy
- p_tqdm from [https://github.com/swansonk14/p_tqdm](https://github.com/swansonk14/p_tqdm).
  which can be installed with `pip install p_tqdm`. This package is used for 
  parallel computing with progress bar.

The simulations are computed in parallel with as many threads as specified by
`cpu_count` default to 4. If running on a stronger computer make sure to
increase this parameter.
