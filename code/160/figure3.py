# replicate_figure3.py — Figure 3 with replicate(); DV passed as exp(lncost)

import os, sys, yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt

# Make replicate() importable (adjust depth if needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate  # your replication function

# -----------------------
# Small DV helper: use exp(log DV) for replicate() (which logs internally)
# -----------------------
def y_from_log(frame: pd.DataFrame, log_col: str, floor: float = 1e-9) -> np.ndarray:
    y = np.exp(frame[log_col].to_numpy())
    # ensure strictly positive for internal log
    y = np.where(y <= 0, floor, y)
    return y

# -----------------------
# CONFIG & LOAD
# -----------------------
with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR     = Path(config["outputdata"])
INPUT_DATA_DIR = Path(config["intermediatedata"])
FIG_DIR        = OUTPUT_DIR / "160" / "01_Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_stata(INPUT_DATA_DIR / "160" / "reg_data.dta")

# sanity check
needed = [
    "fire_id_new","date","state","year","lncost","distanceNM","unitid","mofy",
    "windspeed","slope","tmean","vpdmean","ppt","south","fuelmodel"
]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in reg_data.dta: {missing}")

# -----------------------
# CONTROLS & FIXED EFFECTS
# -----------------------
for v in ["tmean","windspeed","slope","vpdmean","ppt"]:
    if f"{v}_sq" not in df.columns:
        df[f"{v}_sq"] = df[v] ** 2

controls = [
    "tmean","tmean_sq","windspeed","windspeed_sq","slope","slope_sq",
    "vpdmean","vpdmean_sq","ppt","ppt_sq","south"
]

fuel_dum = pd.get_dummies(df["fuelmodel"], prefix="fuelmodel", drop_first=True)

df["state_mofy"] = df["state"].astype(str) + "_" + df["mofy"].astype(int).astype(str)
df["state_year"] = df["state"].astype(str) + "_" + df["year"].astype(int).astype(str)

df["unitid_fe"]     = pd.Categorical(df["unitid"]).codes
df["state_mofy_fe"] = pd.Categorical(df["state_mofy"]).codes
df["state_year_fe"] = pd.Categorical(df["state_year"]).codes

fe_cols = ["unitid_fe","state_mofy_fe","state_year_fe"]

# Core working frame (no 'cost' used anywhere)
base = pd.concat([df[["lncost","distanceNM"] + controls + fe_cols], fuel_dum], axis=1)
base = base.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["lncost","distanceNM"] + controls + fe_cols
)
fuel_dum = fuel_dum.loc[base.index]  # align

# -----------------------
# DEP VAR: levels = exp(lncost) since replicate() logs internally
# -----------------------
y = y_from_log(base, "lncost")

# -----------------------
# (1) BINNED 5 KM (0–5 omitted)
# -----------------------
bins_edges = list(range(0, 55, 5))  # 0,5,...,50
bins = pd.IntervalIndex.from_breaks(bins_edges, closed="right")
dist_cut = pd.cut(base["distanceNM"].clip(upper=49.999), bins=bins)
dummies = pd.get_dummies(dist_cut, prefix="distancebin")

expected_bins = [f"distancebin_{iv}" for iv in bins]
for col in expected_bins:
    if col not in dummies.columns:
        dummies[col] = 0

omit = f"distancebin_{bins[0]}"         # (0,5] omitted
bin_cols = [c for c in expected_bins if c != omit]
dummies = dummies[bin_cols]

X_bin = pd.concat([dummies, base[controls], fuel_dum, base[fe_cols]], axis=1)
X_bin = sm.add_constant(X_bin, prepend=False)
fe_idx_bin = [X_bin.columns.get_loc(c) for c in fe_cols]
bin_interest = bin_cols[0]  # single interest var for replicate()

# -----------------------
# (2) CUBIC
# -----------------------
cubic = base[["distanceNM"]].copy()
cubic["dist"]  = cubic["distanceNM"]
cubic["dist2"] = cubic["distanceNM"]**2
cubic["dist3"] = cubic["distanceNM"]**3
dist_cols = ["dist","dist2","dist3"]

X_cubic = pd.concat([cubic[dist_cols], base[controls], fuel_dum, base[fe_cols]], axis=1)
X_cubic = sm.add_constant(X_cubic, prepend=False)
fe_idx_cubic = [X_cubic.columns.get_loc(c) for c in fe_cols]
cubic_interest = dist_cols[0]

# -----------------------
# (3) LINEAR SPLINE @ 10,20,30,40
# -----------------------
def lspline_columns(x, knots=(10,20,30,40), index=None):
    x = np.asarray(x)
    ks = list(knots)
    segs = {}
    prev = 0.0
    for i, k in enumerate(ks, start=1):
        segs[f"s{i}"] = np.clip(x - prev, 0, k - prev)
        prev = k
    segs[f"s{len(ks)+1}"] = np.clip(x - prev, 0, None)
    return pd.DataFrame(segs, index=index)

spline = lspline_columns(base["distanceNM"], knots=(10,20,30,40), index=base.index)
spline_cols = list(spline.columns)

X_spline = pd.concat([spline, base[controls], fuel_dum, base[fe_cols]], axis=1)
X_spline = sm.add_constant(X_spline, prepend=False)
fe_idx_spline = [X_spline.columns.get_loc(c) for c in fe_cols]
spline_interest = spline_cols[0]

# -----------------------
# RUN THREE REGS (save bundle once to avoid overwrite)
# -----------------------
meta_common = {
    "paper_id": "160",
    "table_id": "F3",
    "panel_identifier": "figure3",
    "model_type": "log-linear",
}

# Binned — SAVE
rep_bin, res_bin = replicate(
    metadata={**meta_common, "comments": "Figure 3 (Binned 5km): omitted 0–5km; controls + FE + fuelmodel."},
    y=y, X=X_bin, interest=bin_interest, fe=fe_idx_bin,
    elasticity=False, output=True, output_dir=str(OUTPUT_DIR), replicated=True
)

# Cubic — NO SAVE (avoid FileExistsError)
rep_cub, res_cub = replicate(
    metadata={**meta_common, "comments": "Figure 3 (Cubic distance)."},
    y=y, X=X_cubic, interest=cubic_interest, fe=fe_idx_cubic,
    elasticity=False, output=False, output_dir=str(OUTPUT_DIR), replicated=True
)

# Spline — NO SAVE
rep_spl, res_spl = replicate(
    metadata={**meta_common, "comments": "Figure 3 (Linear spline at 10,20,30,40)."},
    y=y, X=X_spline, interest=spline_interest, fe=fe_idx_spline,
    elasticity=False, output=False, output_dir=str(OUTPUT_DIR), replicated=True
)

# -----------------------
# PARAM/COV WRAPPERS + FAST PRED
# -----------------------
def _wrap_params_and_cov(result, exog_names):
    names = list(exog_names)
    params = pd.Series(np.asarray(result.params).ravel(), index=names)
    cov = np.asarray(result.cov_params())
    cov = pd.DataFrame(cov, index=names, columns=names)
    return params, cov

def predict_linear_combo(result, X_full, cols, exog_names):
    """
    Predict using only 'cols' (others zero). Returns (est, se) arrays.
    Complexity O(N * k^2) with k=len(cols); no N^2 objects.
    """
    params, cov = _wrap_params_and_cov(result, exog_names)
    cols = list(cols)
    Xk = X_full[cols].to_numpy(dtype=float)
    beta_k = params[cols].to_numpy(dtype=float)
    cov_k = cov.loc[cols, cols].to_numpy(dtype=float)
    est = Xk @ beta_k
    var = np.einsum('ij,jk,ik->i', Xk, cov_k, Xk, optimize=True)
    se = np.sqrt(np.clip(var, 0, None))
    return est, se

# BINNED (step): use all bin dummies to form the step line
bin_exog_names = list(rep_bin.estimator.exog_names)
est_bin, _se_bin = predict_linear_combo(res_bin, X_bin, cols=bin_cols, exog_names=bin_exog_names)

dist_plot = base["distanceNM"].to_numpy()
df_step = pd.DataFrame({"x": dist_plot, "yb": est_bin})
df_step["bin"] = (np.minimum((df_step["x"] // 5) * 5 + 5, 50)).astype(int)
step_vals = df_step.groupby("bin")["yb"].mean().reset_index().sort_values("bin")
bx = [0]
by = [step_vals.loc[step_vals["bin"] == 5, "yb"].mean() if (step_vals["bin"]==5).any() else 0.0]
for _, r in step_vals.iterrows():
    b = int(r["bin"]); v = float(r["yb"])
    if b > 5:
        bx += [b-5, b]
        by += [v, v]
    else:
        bx += [b]
        by += [v]

# SPLINE (line)
spl_exog_names = list(rep_spl.estimator.exog_names)
est_spl, _se_spl = predict_linear_combo(res_spl, X_spline, cols=spline_cols, exog_names=spl_exog_names)
order = np.argsort(dist_plot)
x_obs = dist_plot[order]
y_spl = est_spl[order]

# CUBIC (grid + 95% CI)
xg = np.linspace(0, 50, 201)
D = pd.DataFrame({"dist": xg, "dist2": xg**2, "dist3": xg**3})
cub_exog_names = list(rep_cub.estimator.exog_names)

Xg = pd.DataFrame(0.0, index=D.index, columns=cub_exog_names)
for c in ["dist","dist2","dist3"]:
    if c in Xg.columns:
        Xg[c] = D[c].to_numpy()

y_cub, se_cub = predict_linear_combo(res_cub, Xg, cols=["dist","dist2","dist3"], exog_names=cub_exog_names)
y_lo = y_cub - 1.96 * se_cub
y_hi = y_cub + 1.96 * se_cub

# -----------------------
# PLOT
# -----------------------
fig, ax1 = plt.subplots(figsize=(7.5, 4.0))

# Binned
ax1.step(bx, by, where="post", label="Binned", linewidth=1.5)

# Cubic with ribbon
ax1.plot(xg, y_cub, label="Cubic", linewidth=1.5)
ax1.fill_between(xg, y_lo, y_hi, alpha=0.25, linewidth=0)

# Spline
ax1.plot(x_obs, y_spl, label="Spline", linewidth=1.5)

ax1.set_xlim(0, 50)
ax1.set_xlabel("Nearest home (km)")
ax1.set_ylabel("log(Suppression cost)")

def same_transform(v): return v
sec = ax1.secondary_yaxis("right", functions=(same_transform, same_transform))
sec.set_ylabel("Percent decrease in cost")
sec.set_yticks([-4,-3,-2,-1,0])
sec.set_yticklabels(["98%","95%","86%","63%","0%"])

ax1.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)
ax1.grid(axis="y", alpha=0.3)

outpath = FIG_DIR / "figure3.pdf"
fig.tight_layout()
fig.savefig(outpath, bbox_inches="tight")
print(f"Saved {outpath}")
