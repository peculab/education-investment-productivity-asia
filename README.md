# education-investment-productivity-asia

This repository contains the data pipeline, statistical analysis, and figure generation code for a paper on the relationship between education investment and labor productivity in East and Southeast Asia.

The project asks a cautious empirical question:

> Does higher government expenditure on education translate into higher labor productivity, and if so, is that relationship immediate, delayed, or conditional?

**Latex Paper**

https://www.overleaf.com/read/qzvdvfmcrwqt#471881

The current design does **not** support a strong threshold claim. Instead, the analysis is framed around **conditional and delayed returns to education investment**.

## Repository Structure

```text
education-investment-productivity-asia/
|-- data_csv/
|   |-- GDP per worker.csv
|   |-- Government_expenditure.csv
|   |-- Individuals using the Internet.csv
|   |-- P_Data_Extract_From_Jobs .csv
|   `-- rd_expenditure.csv
|-- run_tokyo_analysis.py
|-- refine_tokyo_analysis.py
|-- output/
|   |-- figures/
|   |-- tables/
|   |-- analysis_summary.md
|   `-- paper_methods_results_draft.md
`-- README.md
```

## What the Scripts Do

### `run_tokyo_analysis.py`

This is the main analysis pipeline. It:

- loads the raw CSV files from `./data_csv`
- reshapes the World Bank indicator files into long panel form
- constructs the East and Southeast Asia sample
- fits the core fixed-effects models
- exports the main tables
- generates Figures 1 to 5
- writes a short `analysis_summary.md`

### `refine_tokyo_analysis.py`

This is the refinement and paper-output script. It:

- rebuilds the panel with the same relative-path data setup
- creates the extended and balanced samples
- fits the refined model suite used for writing
- exports additional regression tables
- generates Figures 6 to 8
- writes a more paper-oriented `paper_methods_results_draft.md`

## Quick Start

Open a terminal in the repository root:

```powershell
PS C:\education-investment-productivity-asia>
```

Run the full analysis:

```powershell
python .\run_tokyo_analysis.py
python .\refine_tokyo_analysis.py
```

Both scripts use **relative paths**, so they should be run from the repository root. The raw data are expected in:

```text
./data_csv/
```

The scripts will create:

```text
./output/figures
./output/tables
./output/analysis_summary.md
./output/paper_methods_results_draft.md
```

## Python Dependencies

The scripts require a standard scientific Python stack:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `statsmodels`

If needed, install them with:

```powershell
pip install pandas numpy matplotlib seaborn statsmodels
```

## Expected Outputs

After both scripts finish, the repository should contain the following figure files in `output/figures`:

- `fig_01_coverage_heatmap.png`
- `fig_02_index_trends.png`
- `fig_03_diagnostic_scatter.png`
- `fig_04_country_mean_scatter.png`
- `fig_05_coefficient_plot.png`
- `fig_06_country_trajectories.png`
- `fig_07_country_facets.png`
- `fig_08_education_robustness.png`

Key tables are written to `output/tables`, including:

- `analysis_sample_panel.csv`
- `analysis_sample_panel_extended.csv`
- `analysis_sample_panel_balanced.csv`
- `model_terms.csv`
- `model_terms_refined.csv`
- `paper_regression_table.csv`

## Data and Sample Design

The current design uses a shared annual panel from **2016 to 2023**.

Why this matters:

- `GDP per worker.csv` provides usable annual observations mainly for `2016-2025` plus `2000`
- this makes the common panel relatively short
- the regional panel is also unevenly observed across countries
- because of that, a threshold regression design was dropped

The project therefore relies on a simpler and more defensible model sequence:

1. same-year two-way fixed effects as a diagnostic benchmark
2. one-year-lagged two-way fixed effects as the primary specification
3. next-year productivity growth as a dynamic robustness check
4. balanced-sample re-estimation as a sensitivity check

## Current Empirical Reading

The main pattern from the current version is:

- the same-year fixed-effects model shows a negative and statistically significant education coefficient
- that negative relationship weakens and becomes statistically inconclusive once education expenditure is lagged
- in the next-year growth model, the sign turns positive but remains statistically inconclusive

In substantive terms, the most defensible interpretation is:

- **not** that education spending clearly harms productivity
- **not** that the data clearly support an immediate linear payoff
- but that returns to education investment appear **delayed, conditional, and hard to detect** when spending is measured as a share of GDP in a short national panel

## How to Read the Results

This repository should be read as an analysis of **national HRD capability conversion**, not just as a simple spending-to-output exercise.

A careful reading is:

- education spending alone may be insufficient to generate immediate productivity gains
- labor-market absorption, digital readiness, and institutional alignment likely matter
- same-year models should be treated cautiously because education expenditure is measured as a percentage of GDP and may reflect denominator effects during downturns

So if you use this repository for writing or presentation, the safest framing is:

> education investment may have conditional and delayed returns, rather than a stable contemporaneous productivity payoff

## Recommended Citation Framing for the Paper

The analysis is best aligned with themes such as:

- national HRD
- education-to-work conversion
- skill mismatch
- delayed returns to education investment
- digital transformation and workforce capability

## Notes

- The scripts are designed for local reruns with the raw CSV files already placed in `data_csv/`.
- The repository currently focuses on reproducible analysis outputs rather than packaging or distribution.
- If you are preparing a blind-review submission, keep the repository private until author-identifying information is removed where necessary.
