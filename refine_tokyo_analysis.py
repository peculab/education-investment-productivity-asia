from pathlib import Path
import re, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'output'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
for p in [FIG, TAB]: p.mkdir(parents=True, exist_ok=True)
FILES = {
    'GDP_y': ('GDP per worker.csv', 0),
    'Edu_x': ('Government_expenditure.csv', 3),
    'Digital_w': ('Individuals using the Internet.csv', 3),
    'Labor_z2': ('P_Data_Extract_From_Jobs .csv', 3),
}
CANDS = ['Brunei Darussalam','Cambodia','China','Hong Kong SAR, China','Indonesia','Japan','Korea, Rep.','Lao PDR','Macao SAR, China','Malaysia','Mongolia','Myanmar','Philippines','Singapore','Thailand','Timor-Leste','Viet Nam']

def load(name, skip, val):
    df = pd.read_csv(ROOT / 'data_csv' / name, skiprows=skip)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df['Country Code'].notna()].copy()
    yrs = [c for c in df.columns if re.match(r'^\d{4}', str(c))]
    df = df[['Country Name','Country Code'] + yrs]
    df = df.melt(id_vars=['Country Name','Country Code'], var_name='Year', value_name=val)
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')[0]
    df = df[df['Year'].notna()].copy()
    df['Year'] = df['Year'].astype(int)
    df[val] = pd.to_numeric(df[val].replace('..', pd.NA), errors='coerce')
    return df

def build():
    m = None
    for val, (name, skip) in FILES.items():
        d = load(name, skip, val)
        m = d if m is None else m.merge(d, on=['Country Name','Country Code','Year'], how='outer')
    return m.sort_values(['Country Name','Year']).reset_index(drop=True)

def prep(df, names):
    d = df[df['Country Name'].isin(names) & df['core']].copy().sort_values(['Country Name','Year'])
    d['ln_GDP_y'] = np.log(d['GDP_y'])
    for c in ['Edu_x','Digital_w','Labor_z2','ln_GDP_y']:
        d[c + '_lag1'] = d.groupby('Country Name')[c].shift(1)
    d['gdp_growth_next'] = d.groupby('Country Name')['ln_GDP_y'].shift(-1) - d['ln_GDP_y']
    d['Edu_within'] = d['Edu_x'] - d.groupby('Country Name')['Edu_x'].transform('mean')
    d['ln_GDP_within'] = d['ln_GDP_y'] - d.groupby('Country Name')['ln_GDP_y'].transform('mean')
    for c in ['GDP_y','Edu_x','Digital_w']:
        d[c + '_idx'] = d[c] / d.groupby('Country Name')[c].transform('first') * 100
    return d.reset_index(drop=True)

def fit_suite(df, label):
    rows = []
    specs = []
    a = df.copy()
    for c in ['Edu_x','Digital_w','Labor_z2']:
        a[c.replace('_x','').replace('_w','').replace('_z2','') + '_c'] = a[c] - a[c].mean()
    specs.append((a, 'ln_GDP_y ~ Edu_c + Digital_c + Labor_c + C(Q("Country Name")) + C(Year)', 'Same-year FE'))
    b = df.dropna(subset=['Edu_x_lag1','Digital_w_lag1','Labor_z2_lag1']).copy()
    b['Edu_c'] = b['Edu_x_lag1'] - b['Edu_x_lag1'].mean(); b['Digital_c'] = b['Digital_w_lag1'] - b['Digital_w_lag1'].mean(); b['Labor_c'] = b['Labor_z2_lag1'] - b['Labor_z2_lag1'].mean()
    specs.append((b, 'ln_GDP_y ~ Edu_c + Digital_c + Labor_c + C(Q("Country Name")) + C(Year)', 'Lagged FE'))
    c = df.dropna(subset=['Edu_x_lag1','Digital_w_lag1','Labor_z2_lag1','gdp_growth_next']).copy()
    c['Edu_c'] = c['Edu_x_lag1'] - c['Edu_x_lag1'].mean(); c['Digital_c'] = c['Digital_w_lag1'] - c['Digital_w_lag1'].mean(); c['Labor_c'] = c['Labor_z2_lag1'] - c['Labor_z2_lag1'].mean()
    specs.append((c, 'gdp_growth_next ~ Edu_c + Digital_c + Labor_c + C(Q("Country Name")) + C(Year)', 'Next-year Growth FE'))
    labels = {'Edu_c':'Education expenditure','Digital_c':'Internet use','Labor_c':'Labor force participation'}
    for dat, formula, model in specs:
        r = smf.ols(formula, data=dat).fit(cov_type='HC3')
        for t, lab in labels.items():
            coef, se = float(r.params[t]), float(r.bse[t])
            rows.append({'sample':label,'model':model,'term':lab,'coef':coef,'p':float(r.pvalues[t]),'lo':coef-1.96*se,'hi':coef+1.96*se,'nobs':int(r.nobs),'r2':float(r.rsquared)})
    return pd.DataFrame(rows)

def coef(df, s, m, t='Education expenditure'):
    return df[(df['sample']==s)&(df['model']==m)&(df['term']==t)].iloc[0]

def fmt(c, p):
    stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
    return f'{c:.3f}{stars} (p={p:.3f})'

sns.set_theme(style='whitegrid', context='talk')
panel = build()
reg = panel[panel['Country Name'].isin(CANDS) & panel['Year'].between(2016, 2023)].copy()
reg['core'] = reg[['GDP_y','Edu_x','Digital_w','Labor_z2']].notna().all(axis=1)
cover = reg.groupby('Country Name')['core'].sum().sort_values(ascending=False)
ext_names = cover[cover >= 6].index.tolist(); bal_names = cover[cover == 8].index.tolist()
ext = prep(reg, ext_names); bal = prep(reg, bal_names)
terms = pd.concat([fit_suite(ext, 'Extended sample'), fit_suite(bal, 'Balanced sample')], ignore_index=True)

ext.to_csv(TAB / 'analysis_sample_panel_extended.csv', index=False)
bal.to_csv(TAB / 'analysis_sample_panel_balanced.csv', index=False)
terms.to_csv(TAB / 'model_terms_refined.csv', index=False)
pd.DataFrame({'Country Name':cover.index,'core_obs':cover.values}).to_csv(TAB / 'country_coverage_refined.csv', index=False)
rows = []
for (s,m), g in terms.groupby(['sample','model']):
    lk = {r['term']:r for _, r in g.iterrows()}
    rows.append({'sample':s,'model':m,'nobs':int(g['nobs'].iloc[0]),'r2':float(g['r2'].iloc[0]),'Education expenditure':fmt(lk['Education expenditure']['coef'], lk['Education expenditure']['p']),'Internet use':fmt(lk['Internet use']['coef'], lk['Internet use']['p']),'Labor force participation':fmt(lk['Labor force participation']['coef'], lk['Labor force participation']['p'])})
pd.DataFrame(rows).to_csv(TAB / 'paper_regression_table.csv', index=False)

fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
for country, g in ext.groupby('Country Name'):
    ax[0].plot(g['Year'], g['GDP_y_idx'], marker='o', linewidth=1.8, alpha=.85, label=country)
    ax[1].plot(g['Year'], g['Edu_x_idx'], marker='o', linewidth=1.8, alpha=.85, label=country)
ax[0].set_title('Country productivity trajectories (2016 = 100)'); ax[1].set_title('Country education expenditure trajectories (2016 = 100)')
ax[0].set_ylabel('Indexed productivity'); ax[1].set_ylabel('Indexed education spending share')
for a in ax: a.set_xlabel('Year'); a.grid(alpha=.2)
handles, labels = ax[1].get_legend_handles_labels(); fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, .5), frameon=False)
fig.tight_layout(); fig.savefig(FIG / 'fig_06_country_trajectories.png', dpi=300, bbox_inches='tight'); plt.close(fig)

def facet_reg(data, color=None, **k):
    sns.regplot(data=data, x='Edu_within', y='ln_GDP_within', scatter_kws={'s':28, 'alpha':.75, 'color':'#4c78a8'}, line_kws={'color':'#d62728','linewidth':1.8}, truncate=False)

g = sns.FacetGrid(ext, col='Country Name', col_wrap=4, height=2.8, sharex=False, sharey=False, despine=False)
g.map_dataframe(facet_reg)
g.set_axis_labels('Education expenditure, demeaned', 'Log productivity, demeaned'); g.set_titles('{col_name}')
for a in g.axes.flat: a.axhline(0, color='gray', linewidth=.6, linestyle='--'); a.axvline(0, color='gray', linewidth=.6, linestyle='--'); a.grid(alpha=.15)
g.fig.suptitle('Within-country education-productivity association by country', y=1.02); g.tight_layout(); g.savefig(FIG / 'fig_07_country_facets.png', dpi=300, bbox_inches='tight'); plt.close(g.fig)

rob = terms[terms['term']=='Education expenditure'].copy(); rob['label'] = rob['sample'] + ' | ' + rob['model']; rob = rob.sort_values(['sample','model']).reset_index(drop=True)
y = np.arange(len(rob))[::-1]; colors = {'Extended sample':'#1f77b4', 'Balanced sample':'#ff7f0e'}
plt.figure(figsize=(9.5, 4.8))
for yy, (_, r) in zip(y, rob.iterrows()):
    c = colors[r['sample']]; plt.plot([r['lo'], r['hi']], [yy, yy], color=c, linewidth=2); plt.scatter(r['coef'], yy, color=c, s=40, zorder=3)
plt.axvline(0, color='gray', linestyle='--', linewidth=1); plt.yticks(y, rob['label']); plt.xlabel('Education expenditure coefficient (95% CI)'); plt.title('Education coefficient robustness across samples and models'); plt.tight_layout(); plt.savefig(FIG / 'fig_08_education_robustness.png', dpi=300, bbox_inches='tight'); plt.close()

s1 = coef(terms, 'Extended sample', 'Same-year FE'); s2 = coef(terms, 'Extended sample', 'Lagged FE'); s3 = coef(terms, 'Extended sample', 'Next-year Growth FE'); s4 = coef(terms, 'Balanced sample', 'Lagged FE')
idx = ext.groupby('Year')[['GDP_y_idx','Edu_x_idx','Digital_w_idx']].median().loc[2023]
summary = f"""# tokyo_data analysis summary\n\n## Revised design\n\n- Shared panel window: 2016-2023\n- Extended sample: {ext['Country Name'].nunique()} countries, {len(ext)} observations\n- Balanced sample: {bal['Country Name'].nunique()} countries, {len(bal)} observations\n- Extended countries: {', '.join(ext_names)}\n- Balanced countries: {', '.join(bal_names)}\n- Threshold regression was dropped because the annual panel is too short and unevenly observed for reliable threshold identification.\n\n## Main findings\n\n- Same-year FE education coefficient: {s1['coef']:.4f} (p = {s1['p']:.4f})\n- Lagged FE education coefficient, extended sample: {s2['coef']:.4f} (p = {s2['p']:.4f})\n- Lagged FE education coefficient, balanced sample: {s4['coef']:.4f} (p = {s4['p']:.4f})\n- Next-year growth education coefficient: {s3['coef']:.4f} (p = {s3['p']:.4f})\n- Median 2023 indices relative to 2016: productivity = {idx['GDP_y_idx']:.1f}, education share = {idx['Edu_x_idx']:.1f}, internet use = {idx['Digital_w_idx']:.1f}.\n\n## Interpretation\n\n- The strong negative same-year coefficient weakens once education expenditure is lagged.\n- The balanced-sample sensitivity check preserves the sign but remains statistically inconclusive with HC3 standard errors.\n- The data support a cautious framing around delayed or conditional returns to education investment, not a strong threshold claim.\n"""
(OUT / 'analysis_summary.md').write_text(summary, encoding='utf-8')

draft = f"""# Paper-Ready Methods and Results Draft\n\n## Methods\nThis study uses a short regional panel assembled from World Bank indicator files included in the tokyo_data project. The dependent variable is GDP per person employed, transformed into natural logarithms for the level models. Education expenditure as a percentage of GDP is the focal predictor, while Internet use and labor force participation are included as contextual covariates. Because the GDP-per-worker source only contains 2000 and 2016-2025 observations, the shared annual panel is effectively limited to 2016-2023.\n\nThe empirical setting is East and Southeast Asia. Seventeen candidate economies were screened. The extended sample retained countries with at least six complete country-year observations for GDP per worker, education expenditure, Internet use, and labor force participation, yielding {ext['Country Name'].nunique()} countries and {len(ext)} observations. A balanced sample requiring full 2016-2023 coverage yielded {bal['Country Name'].nunique()} countries and {len(bal)} observations. Because the concept note's threshold design would be weakly identified under this short and uneven panel, the revised strategy estimates three simpler models: a same-year two-way fixed-effects diagnostic, a one-year-lagged two-way fixed-effects model as the primary specification, and a next-year productivity-growth model. HC3 robust standard errors are used throughout.\n\n## Results\nCoverage diagnostics show that the regional sample is unevenly observed, which justifies the use of extended and balanced samples rather than a threshold model. By 2023, the median country in the extended sample reached index values of {idx['GDP_y_idx']:.1f} for productivity, {idx['Edu_x_idx']:.1f} for education expenditure share, and {idx['Digital_w_idx']:.1f} for Internet use relative to 2016. Internet use rose steadily across the region, while education expenditure as a share of GDP moved much more heterogeneously.\n\nIn the extended sample, the same-year fixed-effects model produces a negative coefficient on education expenditure (b = {s1['coef']:.3f}, p = {s1['p']:.3f}). However, once education expenditure is lagged by one year, the coefficient remains negative but becomes statistically inconclusive (b = {s2['coef']:.3f}, p = {s2['p']:.3f}). The balanced-sample lagged model preserves the negative sign (b = {s4['coef']:.3f}, p = {s4['p']:.3f}), indicating that the directional pattern is not driven only by partially observed countries. At the same time, the lack of precision means the evidence is not strong enough to support a confident negative effect.\n\nThe dynamic growth specification further tempers the negative interpretation. In the extended sample, lagged education expenditure is positively associated with next-year productivity growth, but the estimate is also statistically inconclusive (b = {s3['coef']:.3f}, p = {s3['p']:.3f}). Taken together, the model sequence suggests that the strongest negative result is contemporaneous and likely reflects timing or denominator effects tied to measuring education expenditure as a share of GDP. For the paper, the most defensible framing is therefore conditional and delayed returns to education investment rather than a strong threshold effect or a simple linear productivity payoff.\n"""
(OUT / 'paper_methods_results_draft.md').write_text(draft, encoding='utf-8')
print('Refined outputs saved to', OUT)
