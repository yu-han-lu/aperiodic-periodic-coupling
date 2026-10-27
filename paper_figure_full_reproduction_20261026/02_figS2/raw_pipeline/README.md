# Raw Supplementary Figure 2 pipeline

`python run_raw.py --full` executes the final analysis in two stages:

1. simulate 30-s OU-plus-bandpass inputs at 1,000 Hz, estimate Welch spectra, fit FOOOF 1.1.1 knee mode and IRASA–SPLIT, fit first- and second-degree calibration response surfaces, and combine the second-order parameter map with complete two-coordinate calibration realization residuals;
2. regenerate panels d–f with equal input-setting weights. Theory and held-out evaluation use the same uniformly sampled evaluable settings; one valid held-out realization is sampled within a selected setting, and one complete calibration residual pair is added to the predicted conditional mean.

Registered counts are 20 groups per band, 64 calibration settings × 8 realizations, 128 held-out settings × 4 realizations, and 50,000 equal-setting resamples. The high-gamma strength range is 28–34; the other bands use 2.8–3.8.

`external_data/` contains the registered base signal, IRASA background-residual reference and empirical knee sample required by the generator. They are inputs to the simulation protocol, not fitted outputs from the held-out groups.
