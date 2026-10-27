# Raw Supplementary Figure 4 pipeline

The figure combines two registered calculation layers and the launcher keeps them separate:

- `--observation-model`: the finite-window observation-model population validation around the linearized Hopf resolvent;
- `--exact-b` or `--exact-f`: complete cubic Hopf time-domain trajectories integrated with explicit Heun for the corresponding stable-node or stable-complex panel set.

`python run_raw.py --full` runs both layers. `--max-items` and `--walltime-minutes` only bound resumable exact-nonlinear work in the current invocation; they do not alter the registered condition, trial or numerical settings.
