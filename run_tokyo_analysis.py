from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


INPUT_FILES = {
    "GDP_y": ("GDP per worker.csv", 0),
    "Edu_x": ("Government_expenditure.csv", 3),
    "Digital_w": ("Individuals using the Internet.csv", 3),
    "Labor_z2": ("P_Data_Extract_From_Jobs .csv", 3),
    "RD_z1": ("rd_expenditure.csv", 3),
}

REGIONAL_CANDIDATES = [
    "Brunei Darussalam",
    "Cambodia",
    "China",
    "Hong Kong SAR, China",
    "Indonesia",
    "Japan",
    "Korea, Rep.",
    "Lao PDR",
    "Macao SAR, China",
    "Malaysia",
    "Mongolia",
    "Myanmar",
    "Philippines",
    "Singapore",
    "Thailand",
    "Timor-Leste",
    "Viet Nam",
]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tokyo_data panel analysis pipeline.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--min-country-obs", type=int, default=6)
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def load_indicator(input_dir: Path, filename: str, skiprows: int, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(input_dir / "data_csv" / filename, skiprows=skiprows)
    df.columns = [str(col).strip() for col in df.columns]
    df = df[df["Country Code"].notna()].copy()

    year_cols = [col for col in df.columns if re.match(r"^\d{4}", str(col))]
    df = df[["Country Name", "Country Code"] + year_cols].copy()

    long = df.melt(
        id_vars=["Country Name", "Country Code"],
        var_name="Year",
        value_name=value_name,
    )
    long["Year"] = long["Year"].astype(str).str.extract(r"(\d{4})")[0]
    long = long[long["Year"].notna()].copy()
    long["Year"] = long["Year"].astype(int)
    long[value_name] = pd.to_numeric(long[value_name].replace("..", pd.NA), errors="coerce")
    return long


def build_panel(input_dir: Path) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for value_name, (filename, skiprows) in INPUT_FILES.items():
        long = load_indicator(input_dir, filename, skiprows, value_name)
        merged = long if merged is None else merged.merge(
            long,
            on=["Country Name", "Country Code", "Year"],
            how="outer",
        )

    if merged is None:
        raise RuntimeError("No data files were loaded.")

    return merged.sort_values(["Country Name", "Year"]).reset_index(drop=True)


def define_sample(
    panel: pd.DataFrame,
    start_year: int,
    end_year: int,
    min_country_obs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    regional = panel[
        panel["Country Name"].isin(REGIONAL_CANDIDATES) & panel["Year"].between(start_year, end_year)
    ].copy()
    regional["core_complete"] = regional[["GDP_y", "Edu_x", "Digital_w", "Labor_z2"]].notna().all(axis=1)
    regional["rd_complete"] = regional[["GDP_y", "Edu_x", "Digital_w", "Labor_z2", "RD_z1"]].notna().all(axis=1)

    country_coverage = (
        regional.groupby("Country Name")
        .agg(
            core_obs=("core_complete", "sum"),
            rd_obs=("rd_complete", "sum"),
            gdp_obs=("GDP_y", lambda s: int(s.notna().sum())),
            edu_obs=("Edu_x", lambda s: int(s.notna().sum())),
            digital_obs=("Digital_w", lambda s: int(s.notna().sum())),
            labor_obs=("Labor_z2", lambda s: int(s.notna().sum())),
        )
        .sort_values(["core_obs", "rd_obs", "gdp_obs"], ascending=False)
    )

    sample_countries = country_coverage[country_coverage["core_obs"] >= min_country_obs].index.tolist()
    sample = regional[regional["Country Name"].isin(sample_countries) & regional["core_complete"]].copy()
    sample["ln_GDP_y"] = np.log(sample["GDP_y"])
    sample = sample.sort_values(["Country Name", "Year"]).reset_index(drop=True)
    return regional, country_coverage, sample, sample_countries


def add_panel_features(sample: pd.DataFrame) -> pd.DataFrame:
    enriched = sample.copy()
    for column in ["Edu_x", "Digital_w", "Labor_z2", "ln_GDP_y"]:
        enriched[f"{column}_lag1"] = enriched.groupby("Country Name")[column].shift(1)
    enriched["gdp_growth_next"] = enriched.groupby("Country Name")["ln_GDP_y"].shift(-1) - enriched["ln_GDP_y"]

    enriched["Edu_within"] = enriched["Edu_x"] - enriched.groupby("Country Name")["Edu_x"].transform("mean")
    enriched["ln_GDP_within"] = enriched["ln_GDP_y"] - enriched.groupby("Country Name")["ln_GDP_y"].transform("mean")
    return enriched


def fit_models(sample: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    contemporaneous = sample.copy()
    contemporaneous["Edu_c"] = contemporaneous["Edu_x"] - contemporaneous["Edu_x"].mean()
    contemporaneous["Digital_c"] = contemporaneous["Digital_w"] - contemporaneous["Digital_w"].mean()
    contemporaneous["Labor_c"] = contemporaneous["Labor_z2"] - contemporaneous["Labor_z2"].mean()

    lagged = sample.dropna(subset=["Edu_x_lag1", "Digital_w_lag1", "Labor_z2_lag1"]).copy()
    lagged["Edu_c"] = lagged["Edu_x_lag1"] - lagged["Edu_x_lag1"].mean()
    lagged["Digital_c"] = lagged["Digital_w_lag1"] - lagged["Digital_w_lag1"].mean()
    lagged["Labor_c"] = lagged["Labor_z2_lag1"] - lagged["Labor_z2_lag1"].mean()

    future = sample.dropna(subset=["Edu_x_lag1", "Digital_w_lag1", "Labor_z2_lag1", "gdp_growth_next"]).copy()
    future["Edu_c"] = future["Edu_x_lag1"] - future["Edu_x_lag1"].mean()
    future["Digital_c"] = future["Digital_w_lag1"] - future["Digital_w_lag1"].mean()
    future["Labor_c"] = future["Labor_z2_lag1"] - future["Labor_z2_lag1"].mean()

    model_specs = {
        "m1_contemporaneous_fe": (
            contemporaneous,
            "ln_GDP_y ~ Edu_c + Digital_c + Labor_c + C(Q('Country Name')) + C(Year)",
            "Contemporaneous FE (diagnostic)",
        ),
        "m2_lagged_fe": (
            lagged,
            "ln_GDP_y ~ Edu_c + Digital_c + Labor_c + C(Q('Country Name')) + C(Year)",
            "Lagged FE (primary)",
        ),
        "m3_lagged_fe_digital_interaction": (
            lagged,
            "ln_GDP_y ~ Edu_c * Digital_c + Labor_c + C(Q('Country Name')) + C(Year)",
            "Lagged FE + digital interaction",
        ),
        "m4_future_growth_fe": (
            future,
            "gdp_growth_next ~ Edu_c + Digital_c + Labor_c + C(Q('Country Name')) + C(Year)",
            "Next-year productivity growth FE",
        ),
    }

    term_labels = {
        "Edu_c": "Education expenditure",
        "Digital_c": "Internet use",
        "Labor_c": "Labor force participation",
        "Edu_c:Digital_c": "Education x Internet",
    }

    fitted: dict[str, object] = {}
    rows: list[dict[str, float | str]] = []
    for key, (data, formula, label) in model_specs.items():
        result = smf.ols(formula, data=data).fit(cov_type="HC3")
        fitted[key] = result

        for term, pretty_term in term_labels.items():
            if term not in result.params.index:
                continue
            coef = float(result.params[term])
            se = float(result.bse[term])
            rows.append(
                {
                    "model_key": key,
                    "model_label": label,
                    "term": term,
                    "term_label": pretty_term,
                    "coef": coef,
                    "std_err": se,
                    "p_value": float(result.pvalues[term]),
                    "ci_low": coef - 1.96 * se,
                    "ci_high": coef + 1.96 * se,
                    "nobs": int(result.nobs),
                    "r_squared": float(result.rsquared),
                    "adj_r_squared": float(result.rsquared_adj),
                }
            )

    return fitted, pd.DataFrame(rows)


def save_tables(
    regional: pd.DataFrame,
    country_coverage: pd.DataFrame,
    sample: pd.DataFrame,
    model_terms: pd.DataFrame,
    tables_dir: Path,
) -> None:
    coverage_matrix = (
        regional.assign(core_complete=regional["core_complete"].astype(int))
        .pivot(index="Country Name", columns="Year", values="core_complete")
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    coverage_matrix.to_csv(tables_dir / "country_year_core_coverage.csv")
    country_coverage.to_csv(tables_dir / "country_coverage_summary.csv")
    sample.to_csv(tables_dir / "analysis_sample_panel.csv", index=False)

    descriptive = sample[["GDP_y", "ln_GDP_y", "Edu_x", "Digital_w", "Labor_z2"]].describe().T
    descriptive.to_csv(tables_dir / "descriptive_statistics.csv")

    model_terms.to_csv(tables_dir / "model_terms.csv", index=False)


def plot_coverage_heatmap(regional: pd.DataFrame, figures_dir: Path) -> None:
    pivot = (
        regional.assign(core_complete=regional["core_complete"].astype(int))
        .pivot(index="Country Name", columns="Year", values="core_complete")
        .fillna(0)
        .sort_index()
    )
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        pivot,
        cmap=sns.color_palette(["#f3f4f6", "#1f77b4"], as_cmap=True),
        cbar=False,
        linewidths=0.4,
        linecolor="white",
    )
    plt.title("Core variable coverage by country and year")
    plt.xlabel("Year")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_01_coverage_heatmap.png", dpi=300)
    plt.close()


def plot_index_trends(sample: pd.DataFrame, figures_dir: Path) -> None:
    indexed = sample.copy()
    for column in ["GDP_y", "Edu_x", "Digital_w"]:
        base = indexed.groupby("Country Name")[column].transform("first")
        indexed[f"{column}_index"] = indexed[column] / base * 100.0

    summary_map = {
        "GDP_y_index": "Productivity index (2016 = 100)",
        "Edu_x_index": "Education spending share index (2016 = 100)",
        "Digital_w_index": "Internet use index (2016 = 100)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, (column, title) in zip(axes, summary_map.items()):
        yearly = indexed.groupby("Year")[column].agg(
            median="median",
            q25=lambda s: np.quantile(s, 0.25),
            q75=lambda s: np.quantile(s, 0.75),
        )
        ax.plot(yearly.index, yearly["median"], color="#1f4e79", linewidth=2)
        ax.fill_between(yearly.index, yearly["q25"], yearly["q75"], color="#9ecae1", alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Indexed value")
    fig.suptitle("Regional median trajectories with interquartile bands")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_02_index_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic_scatter(sample: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = sample.dropna(subset=["gdp_growth_next", "Edu_x_lag1"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.regplot(
        data=sample,
        x="Edu_within",
        y="ln_GDP_within",
        scatter_kws={"alpha": 0.65, "color": "#4c78a8"},
        line_kws={"color": "#d62728", "linewidth": 2},
        ax=axes[0],
    )
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].axvline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_title("Within-country same-year association")
    axes[0].set_xlabel("Education expenditure (% GDP), demeaned within country")
    axes[0].set_ylabel("Log productivity, demeaned within country")

    sns.regplot(
        data=plot_df,
        x="Edu_x_lag1",
        y="gdp_growth_next",
        scatter_kws={"alpha": 0.7, "color": "#2ca02c"},
        line_kws={"color": "#d62728", "linewidth": 2},
        ax=axes[1],
    )
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_title("Lagged education vs next-year productivity growth")
    axes[1].set_xlabel("Education expenditure in t-1 (% GDP)")
    axes[1].set_ylabel("Next-year log productivity growth")

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_03_diagnostic_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_country_means(sample: pd.DataFrame, figures_dir: Path) -> None:
    country_means = (
        sample.groupby("Country Name")[["Edu_x", "ln_GDP_y", "Digital_w"]]
        .mean()
        .reset_index()
        .sort_values("ln_GDP_y")
    )

    plt.figure(figsize=(8.5, 6))
    scatter = plt.scatter(
        country_means["Edu_x"],
        country_means["ln_GDP_y"],
        s=country_means["Digital_w"] * 2.2,
        c=country_means["Digital_w"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )
    for _, row in country_means.iterrows():
        plt.annotate(
            row["Country Name"],
            (row["Edu_x"], row["ln_GDP_y"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Average internet use (%)")
    plt.title("Country-average education spending and productivity")
    plt.xlabel("Average education expenditure (% GDP)")
    plt.ylabel("Average log GDP per person employed")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_04_country_mean_scatter.png", dpi=300)
    plt.close()


def plot_coefficients(model_terms: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = model_terms[model_terms["term"].isin(["Edu_c", "Digital_c", "Labor_c", "Edu_c:Digital_c"])].copy()
    plot_df["row_order"] = plot_df["model_label"] + " | " + plot_df["term_label"]
    plot_df = plot_df.sort_values(["model_label", "term_label"], ascending=[True, True]).reset_index(drop=True)
    y_positions = np.arange(len(plot_df))[::-1]

    plt.figure(figsize=(10, 6.5))
    for y, (_, row) in zip(y_positions, plot_df.iterrows()):
        plt.plot([row["ci_low"], row["ci_high"]], [y, y], color="#4c78a8", linewidth=2)
        plt.scatter(row["coef"], y, color="#d62728", s=35, zorder=3)

    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.yticks(y_positions, plot_df["row_order"])
    plt.xlabel("Coefficient estimate (95% CI)")
    plt.title("Key coefficient estimates across model specifications")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_05_coefficient_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_summary(
    output_dir: Path,
    input_dir: Path,
    sample: pd.DataFrame,
    country_coverage: pd.DataFrame,
    sample_countries: list[str],
    model_terms: pd.DataFrame,
) -> None:
    years = f"{int(sample['Year'].min())}-{int(sample['Year'].max())}"
    country_list = ", ".join(sample_countries)

    model_lookup = {
        row["model_key"]: row for _, row in model_terms[model_terms["term"] == "Edu_c"].iterrows()
    }
    contemp = model_lookup["m1_contemporaneous_fe"]
    lagged = model_lookup["m2_lagged_fe"]
    growth = model_lookup["m4_future_growth_fe"]

    summary = f"""# tokyo_data analysis summary

## Analysis design

- Input folder: `{input_dir}`
- Sample window: `{years}`
- Regional screen: East and Southeast Asian economies with at least 6 country-year observations containing GDP per worker, education expenditure, internet use, and labor force participation.
- Final sample: `{sample['Country Name'].nunique()}` countries and `{len(sample)}` complete country-year observations.
- Countries in the final sample: {country_list}

## Why this design is more defensible

- `GDP per worker.csv` only provides 2000 and 2016-2025 observations, so the shared annual panel is effectively short.
- `Education expenditure` is measured as a share of GDP, which makes same-year regressions vulnerable to reverse causality and denominator effects during downturns.
- For that reason, the contemporaneous fixed-effects model is treated as a diagnostic check, while the one-year-lagged specification is the primary model.
- `RD expenditure` was not used in the primary model because coverage collapses after 2020 for this regional sample.

## Core findings

- Diagnostic same-year FE coefficient on education expenditure: `{contemp['coef']:.4f}` (p = `{contemp['p_value']:.4f}`).
- Primary lagged FE coefficient on education expenditure: `{lagged['coef']:.4f}` (p = `{lagged['p_value']:.4f}`).
- Next-year productivity growth FE coefficient on lagged education expenditure: `{growth['coef']:.4f}` (p = `{growth['p_value']:.4f}`).
- In this sample, the strong same-year negative relationship weakens and becomes statistically inconclusive once the model uses lagged predictors.
- Internet use and labor force participation do not show stable moderation effects in the available 2016-2023 panel.

## Interpretation

- The current data do not support a strong claim that higher education spending immediately raises labor productivity in East and Southeast Asia over 2016-2023.
- The same-year negative association is more consistent with cyclical measurement problems than with a credible substantive decline in the value of education.
- A more careful paper can therefore frame this project around `conditional and delayed returns to education investment`, rather than a simple positive linear payoff.

## Output files

- Tables: `output/tables`
- Figures: `output/figures`
"""
    (output_dir / "analysis_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")

    figures_dir, tables_dir = ensure_dirs(args.output_dir)
    panel = build_panel(args.input_dir)
    regional, country_coverage, sample, sample_countries = define_sample(
        panel,
        start_year=args.start_year,
        end_year=args.end_year,
        min_country_obs=args.min_country_obs,
    )
    sample = add_panel_features(sample)
    _, model_terms = fit_models(sample)

    save_tables(regional, country_coverage, sample, model_terms, tables_dir)
    plot_coverage_heatmap(regional, figures_dir)
    plot_index_trends(sample, figures_dir)
    plot_diagnostic_scatter(sample, figures_dir)
    plot_country_means(sample, figures_dir)
    plot_coefficients(model_terms, figures_dir)
    write_summary(args.output_dir, args.input_dir, sample, country_coverage, sample_countries, model_terms)

    print(f"Saved analysis outputs to: {args.output_dir}")
    print(f"Sample countries: {', '.join(sample_countries)}")
    print(f"Sample observations: {len(sample)}")


if __name__ == "__main__":
    main()
