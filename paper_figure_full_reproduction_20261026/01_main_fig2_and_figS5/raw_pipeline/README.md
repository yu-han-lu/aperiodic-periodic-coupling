# Raw whole-brain pipeline

`run_raw.py` invokes the archived, registered scripts with portable package-relative paths. Run `python run_raw.py --check` first. The full commands are listed in the archive-level README.

The accepted branches are kept distinct:

- Fig. 2b,c: complete 55-area threshold-linear network, explicit nonlinear Heun, 1,000 stable-node conditions plus 100 zero-input-correlation ensembles.
- Fig. 2f,g: complete 55-area threshold-linear network, explicit nonlinear Heun, 1,000 stable-complex conditions; Fig. 2f subsequently uses the registered feature-versus-input-parameter first-order theory.
- Fig. 2i: first regenerate the unchanged version-6 held-out nonlinear Heun features and its two-node extension, then run the accepted matched-condition nonlinear Heun calibration, single-realization `Ce`, and absolute `delta r` log-axis analysis.
- Fig. 2j: regenerate the frozen 1–5-node deletion designs, complete nonlinear Heun calibration/held-out states and first-order maps; in the accepted final rerun the two held-out phases remain separate through feature extraction, correlations and deletion effects are averaged after feature extraction, and the theory therefore adds single-realization `Ce` rather than `Ce/2`.
- Fig. S5: registered complete threshold-linear 9×9 time-domain generator. Its archived recurrence is forward Euler; this is explicitly not changed to Heun in the reproduction source.

Frozen condition tables and design arrays are inputs, not fitted outcomes. Cached Experiment-2 forcing files are bundled because they preserve the registered finite-record realizations and seeds.
