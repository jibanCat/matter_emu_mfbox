# Multi-fidelity Emulation for Cosmological Simulations from Multiple Boxsizes using a Graphical Gaussian Process

The code is derived from the multi-fidelity emulator code reported in

> M-F Ho, S Bird, and C Shelton. TBD

including the matter power spectrum data (z=0) to reproduce the multi-fidelity emulator trained with 50 low-fidelity simulations and 3 high-fidelity simulations.

Acknowledgement: We thank Dr Simon Mak and Irene Ji from https://arxiv.org/abs/2108.00306 for providing the code for deep graphical GP model. 

Requirements:
- Python 3.6+
- numpy
- scipy
- GPy
- pyDOE
- emukit


Email me if there's any issues about versions: mho026-at-ucr.edu


## Where can I get the power spectrum data?

Simulations are run with [MP-Gadget code](https://github.com/MP-Gadget/MP-Gadget/).

A simulation submission file generator is here: github.com/jibanCat/SimulationRunnerDM.
I used this generator to prepare the training/testing data in this repo.
