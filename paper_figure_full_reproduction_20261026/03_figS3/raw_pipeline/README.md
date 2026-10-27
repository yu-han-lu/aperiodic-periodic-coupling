# Raw Supplementary Figure 3 pipeline

Run `python run_raw.py --full`. This invokes the registered linear damped-oscillator generator and finite-window observation model. It preserves the 0.25-ms step, 2.048-s burn-in, 4.096-s Hann-Welch segments, calibration/held-out phase split, input-condition tables and feature extractor.

The generated tables are written below `project/experiments/pap_theory_validation/outputs/two_dimensional_ode_validation/damped/`.
