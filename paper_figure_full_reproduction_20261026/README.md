# Reproduction code: Main Figure 2 and Supplementary Figures 2–5

This archive supports two distinct, self-contained workflows for Main Fig. 2 and Figs. S2–S5:

1. **source data → figure**: redraw every final panel from its locked panel-level source data;
2. **raw simulation/estimation → data**: rerun the registered dynamical simulation or estimator-validation code that generated the analysis tables.

The two workflows are deliberately separate. Plotting never silently reruns or refits a model, and raw-data generation never overwrites the locked paper source data.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python run_all.py
python verify_outputs.py
```

`requirements-lock.txt` records the exact versions used for the packaged checks; `requirements.txt` gives compatible lower bounds. The FOOOF 1.1.1 package needed by Fig. S2 is also bundled in that raw pipeline.

To verify that the raw pipelines, model files and Python imports are present without starting a large simulation:

```bash
python run_raw.py main --check
python run_raw.py s2 --check
python run_raw.py s3 --check
python run_raw.py s4 --check
```

Each panel can also be run independently, for example:

```bash
python 01_main_fig2_and_figS5/fig2f/plot.py
python 02_figS2/d/plot.py
python 03_figS3/f1/plot.py
```

SVG, PDF and 300-dpi PNG files are written into the corresponding panel's `output/` directory. `run_all.py --only fig2f` restricts execution to matching paths.

## Directory organization

- `01_main_fig2_and_figS5/`: Main Fig. 2a–j and Fig. S5.
- `02_figS2/`: Fig. S2a, b1–b4, c, d, e and f.
- `03_figS3/`: Fig. S3a, b, c, d1–d3, e1–e2, f1 and f2 (linear damped oscillator).
- `04_figS4/`: Fig. S4a, b, c, d1–d3, e1–e2, f1 and f2 (cubic Hopf normal form).

Within these folders, each subdirectory corresponds to one paper panel or a tightly coupled panel group. `data/` contains the exact panel-level source data; `plot.py` is the executable entry point.

Each of the four top-level figure folders also contains `raw_pipeline/`. These directories contain the registered simulation/estimation sources, model files, frozen designs and portable launchers. Generated raw outputs stay inside the corresponding `raw_pipeline/` directory.

## Workflow A: source data to figure

Run all panels with `python run_all.py`, or run one `plot.py`. This route uses the final, locked source tables and performs no refitting, resampling or condition selection.

## Workflow B: raw simulation or estimator output to data

Full runs are opt-in because several require many CPU-hours and substantial memory. Commands are resumable where the original code was resumable.

```bash
# Main Fig. 2b,c: stable-node and resting-state nonlinear Heun runs
python run_raw.py main --fig2bc --workers 4

# Main Fig. 2f,g: stable-complex nonlinear Heun plus parameter first-order theory
python run_raw.py main --fig2fg --workers 4

# Main Fig. 2i and 2j: accepted deletion analyses
python run_raw.py main --fig2i --workers 4
python run_raw.py main --fig2j --workers 4

# Fig. S5: registered 9×9 full-network direct simulation
python run_raw.py main --figS5

# Fig. S2: OU + oscillation simulation, Welch, FOOOF/IRASA–SPLIT,
# second-order-plus-realization prediction, then equal-setting weighting
python run_raw.py s2 --full

# Fig. S3: linear damped-oscillator observation-model data
python run_raw.py s3 --full

# Fig. S4: linearized finite-window observation model plus exact nonlinear Hopf rerun
python run_raw.py s4 --full
```

For a bounded resumable Fig. S4 nonlinear test, use, for example, `python run_raw.py s4 --exact-f --max-items 1`. This changes only how many registered work items are completed in that invocation; it does not change the model or accepted full-run settings.

### Raw output locations

| Paper panels | Generated data |
|---|---|
| Fig. 2b,c | `01_main_fig2_and_figS5/raw_pipeline/workspace/outputs/wholebrain_exp1_resting_heun_20260909/experiment1_results_heun.csv` and `resting_zero_input_results_heun.csv` |
| Fig. 2f,g | `01_main_fig2_and_figS5/raw_pipeline/workspace/outputs/full_network_heun_20260908_retry/results_full.csv`, `source_data_condition_level.csv`, and `01_main_fig2_and_figS5/raw_pipeline/rerun/outputs/first_order_rerun_20260914/experiment2_first_order_results.csv` |
| Fig. 2i | `01_main_fig2_and_figS5/raw_pipeline/workspace/outputs/fig2i_fig2j_matched_single_realization_20261008/fig2i_matched_calibration_Ce/condition_level_results.csv` and `network_aggregate_results.csv` |
| Fig. 2j | `01_main_fig2_and_figS5/raw_pipeline/workspace/outputs/fig2i_fig2j_matched_single_realization_20261008/fig2j_single_realization_Ce/raw_units.csv.gz` and `plot_units_absolute_delta_r.csv.gz` |
| Fig. S5 | `01_main_fig2_and_figS5/raw_pipeline/project/experiments/pap_theory_validation/outputs/nonlinear_resimulations_exact/wholebrain_9x9/` |
| Fig. S2 | `02_figS2/raw_pipeline/outputs/broad_total_covariance/` and `equal_setting_weights/` |
| Fig. S3 | `03_figS3/raw_pipeline/project/experiments/pap_theory_validation/outputs/two_dimensional_ode_validation/damped/` |
| Fig. S4 | `04_figS4/raw_pipeline/project/experiments/pap_theory_validation/outputs/two_dimensional_ode_validation/hopf/` and `nonlinear_resimulations_exact/hopf/` |

The locked `data/` tables remain the authoritative inputs for exact paper redraws. To compare a fresh rerun against them, use identifiers such as condition, phase, trial, deletion pattern, input area and output area rather than row order alone.

## Locked final versions

The following panels deliberately use the most recent accepted analyses rather than older files that remain elsewhere in the project history.

- **Main Fig. 2c:** overlapping theory/simulation correlation histograms with horizontal boxplots; x range −0.2 to 0.2. The input is the final stable-node Heun result.
- **Main Fig. 2f:** the final stable-complex first-order candidate plotted in the positive-exponent convention, `chi=-A`. This is a sign conversion of both axes only; no simulation or fit was rerun.
- **Main Fig. 2g:** overlapping theory/simulation exponent-correlation histograms with horizontal boxplots; x range −0.6 to 0.6.
- **Main Fig. 2i:** the accepted matched-calibration, single-realization-`Ce`, absolute-delta-r log–log panel, including two-node deletions.
- **Main Fig. 2j:** the accepted single-realization-`Ce` direct/non-direct comparison for deletion of one to five nodes.
- **Fig. S2d–f:** the final **equal input-setting weight** analysis. For each group and estimator, theory and held-out mixtures use the same uniformly sampled evaluable-setting indices. Within a selected setting, one valid held-out realization is sampled uniformly; the theory combines the calibration-fitted second-order feature map with one complete calibration residual pair. These panels are not the earlier valid-realization-count-weighted version.

## What the default run does

The default run is a deterministic **figure reproduction** run. It starts from source tables saved after the registered simulation/estimation pipeline and redraws the panels without refitting, resampling, filtering or changing the statistics. This design makes the complete archive runnable on an ordinary computer and preserves the exact final points.

Representative trajectories and spectra in Figs. S3 and S4 are recomputed with the explicit Heun method at `dt=0.25 ms`; the plotting inputs for the large condition ensembles are frozen source tables from the corresponding completed simulations.

## Scientific provenance by panel

### Main Fig. 2 and Fig. S5

- **2a:** 55-area marmoset FLN matrix and highlighted example areas from the model source data.
- **2b:** first-order stable-node network prediction versus the exact nonlinear full-network Heun simulation, 1,000 conditions.
- **2c:** final zero-input-correlation ensemble correlations from theory and nonlinear Heun simulation.
- **2d:** local PG critical eigenvalue sweep from the registered two-stage scan.
- **2e1,e2:** aperiodic input and representative stable-complex whole-brain output spectrum.
- **2f:** parameter first-order prediction versus exact nonlinear full-network Heun simulation; the archived values are shown as `corr(P,chi)`.
- **2g:** final stable-complex theory and exact nonlinear Heun ensemble-correlation distributions in the exponent convention.
- **2h1,h2:** intact FLN matrix and V1-disconnected matrix.
- **2i:** absolute mean deletion effect, theory versus exact nonlinear held-out simulation, on logarithmic axes.
- **2j:** direct versus non-direct absolute deletion effects and the registered FDR-adjusted sign-flip significance marks.
- **S5:** exact nonlinear whole-brain 9-by-9 input/output validation matrix.

### Fig. S2

- **a:** representative held-out high-gamma source log PSD and FOOOF components.
- **b1,b2:** local first-order estimator influence weights. **b3,b4:** signed frequency-resolved covariance contributions calculated from the locked calibration spectral-error covariance.
- **c:** held-out high-gamma estimates and total-covariance theory contours.
- **d–f:** equal-setting-weight final data described above.

### Figs. S3 and S4

- **a, e1,e2:** deterministic representative input/output traces integrated with Heun and converted to Welch PSDs.
- **b, f1:** final 1,000-condition source tables. Both axes are converted from signed log–log slope `A` to positive exponent `chi=-A`.
- **c:** eigenvalues calculated directly from the stated Jacobian.
- **d1–d3:** deterministic phase-plane trajectories calculated directly from the linear oscillator or cubic Hopf equations.
- **f2:** final representative held-out trial cloud; the vertical coordinate is converted to `chi=-A`.

## Reproduction boundary

This archive distinguishes three operations that must not be conflated:

1. linearizing the dynamical state equations to form a Jacobian/transfer function;
2. taking a first- or second-order expansion of extracted features with respect to input-generating parameters;
3. evaluating the full trial-wise feature map or integrating the nonlinear equations.

The bundled source tables preserve which operation generated each paper panel. The plotting scripts never relabel a transfer-function calculation as a parameter first-order test, and they do not select conditions by final fit quality.

The default command intentionally remains the lightweight source-data redraw. Raw pipelines are included and executable but must be requested explicitly. No launcher changes the registered model, input distributions, seeds, time step, burn-in, analysis window, Welch settings, feature extraction, calibration/held-out split, validity masks, Taylor order or statistical aggregation.

One historical distinction is retained rather than hidden: the Fig. S5 9×9 generator in `nonlinear_resimulations_exact.py` is the registered complete threshold-linear time-domain implementation used for that source table, but its recurrence is forward Euler. The accepted Main Fig. 2b,c,f,g,i,j full-network reruns use explicit Heun. The package does not relabel the Fig. S5 generator as Heun or replace it with a different integrator.
