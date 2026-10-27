# Aperiodic and periodic neural activity are independent by default and couple with circuit state

Project code and processed data for separating aperiodic and periodic activity in human stereo-EEG (sEEG) and testing when they couple.

---

# Summary

**Aperiodic and periodic features are largely uncoupled by default, but can become coupled through strengthened local inhibitory feedback.**

Oscillatory peaks and aperiodic backgrounds are widely used to interpret brain function, yet the physiological meaning of their covariation remains unresolved. Stereo-EEG recordings and simulations show that FOOOF spectral fitting can spuriously couple peak frequency and aperiodic exponent. After accounting for this estimation bias, there is no consistent population-level coupling across 3,245 contacts in 61 patients.

Whole-brain modelling predicts weak coupling from uncorrelated input features at rest and after connectivity disruption, and identifies strengthened local inhibitory feedback as a mechanism for coupling via damped oscillatory responses to aperiodic inputs. Three human studies—stimulation, thermocoagulation, and seizure recordings—support these predictions, including increased beta-band oscillation–exponent coupling during seizures.

---

# Data

`./simulation/data/` contains OU simulations with added oscillations used to show that FOOOF spectral fitting can spuriously couple peak frequency and aperiodic exponent.

`./empirical/data/` contains processed, contact-level sEEG outputs used to reproduce the paper figures: awake coupling after this estimation bias is accounted for (3,245 contacts from 61 patients), and the three follow-up studies that test when coupling emerges (stimulation, thermocoagulation, and seizures). Raw voltage recordings are not included.

---

# Code

`./simulation/code/` generates and plots the simulation figures (`*_gen.m` creates signals; IRASA/SPLIT/FOOOF scripts fit spectra; `fig1d_result.m` loads saved data and plots).

`./empirical/code/` reproduces the empirical figures from processed sEEG outputs (`fig1` awake coupling, `fig3` stimulation, `fig4` ictal state, `fig5` CCEP). The coupling pipeline is shown in `fig1`; later scripts (`fig3`–`fig5`) reuse those estimates and do not repeat the coupling code.

`./tools/` contains shared helpers, including the FieldTrip IRASA wrapper (`IRASA_hz.m`).

FieldTrip is required for IRASA. Python dependencies are `numpy`, `scipy`, `h5py`, `pwlf`, `GPyOpt`, and `fooof`.

---

# paper_figure_full_reproduction_20261026

This folder provides the complete reproduction code for the paper’s Main Figure 2 and Supplementary Figures S2–S5, allowing both redrawing of all panels from locked source data and rerunning of the original simulations and estimators that generated those data.