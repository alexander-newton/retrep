import os, sys, yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Make replicate() importable (adjust if your path depth differs)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -----------------------
# CONFIG & LOAD
# -----------------------
with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR     = config["outputdata"]
INPUT_DATA_DIR = config["intermediatedata"]

# Path to Stata data
filepath = os.path.join(INPUT_DATA_DIR, "160", "reg_data.dta")
df = pd.read_stata(filepath)

# Sanity check: expected columns
needed = [
    "fire_id_new","date","state","year","cost","lncost","distanceNM","unitid","mofy",
    "windspeed","slope","tmean","vpdmean","ppt","south","fuelmodel"
]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in reg_data.dta: {missing}")

# -----------------------
# SPEC: Table 1, Column (2)
# -----------------------

# Distance bins: 0–10 (omitted baseline), report 10–20, 20–30, 30–40, 40–50
bins = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50)], closed="right")
dist = pd.cut(df["distanceNM"], bins=bins)
dummies = pd.get_dummies(dist, prefix="distancebin")

# Ensure expected bin columns exist even if some bins are empty
expected_bins = [
    "distancebin_(10, 20]", "distancebin_(20, 30]",
    "distancebin_(30, 40]", "distancebin_(40, 50]"
]
for col in expected_bins:
    if col not in dummies.columns:
        dummies[col] = 0
dummies = dummies[expected_bins]

# Quadratic controls + south (as in paper)
for v in ["tmean","windspeed","slope","vpdmean","ppt"]:
    df[f"{v}_sq"] = df[v] ** 2

controls = [
    "tmean","tmean_sq","windspeed","windspeed_sq","slope","slope_sq",
    "vpdmean","vpdmean_sq","ppt","ppt_sq","south"
]

# Fuel model dummies (like adding fuelmodel FE)
fuel_dum = pd.get_dummies(df["fuelmodel"], prefix="fuelmodel", drop_first=True)

# Fixed effects: unit, state×mofy, state×year
df["state_mofy"] = df["state"].astype(str) + "_" + df["mofy"].astype(int).astype(str)
df["state_year"] = df["state"].astype(str) + "_" + df["year"].astype(int).astype(str)

df["unitid_fe"]     = pd.Categorical(df["unitid"]).codes
df["state_mofy_fe"] = pd.Categorical(df["state_mofy"]).codes
df["state_year_fe"] = pd.Categorical(df["state_year"]).codes

# Build working frame (drop rows missing essentials; keep alignment)
core_cols = ["cost","lncost"] + controls + ["unitid_fe","state_mofy_fe","state_year_fe"]
work = pd.concat([df[core_cols], dummies, fuel_dum], axis=1)
work = work.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["cost"] + controls + ["unitid_fe","state_mofy_fe","state_year_fe"]
)

# Dependent variable in levels
y = work["cost"].values

# X = distance-bin dummies + controls + fuel dummies + FE placeholders
X = pd.concat(
    [
        work[expected_bins],
        work[controls],
        fuel_dum.loc[work.index],  # align indices
        work[["unitid_fe","state_mofy_fe","state_year_fe"]],
    ],
    axis=1,
)

# Add constant
X = sm.add_constant(X, prepend=False)

# FE indices (columns to absorb)
fe_cols = ["unitid_fe","state_mofy_fe","state_year_fe"]
fe_idx = [X.columns.get_loc(c) for c in fe_cols]

# Variables of interest (the distance-bin coefficients reported in Table 1)
interest = expected_bins

# -----------------------
# RUN
# -----------------------
metadata = {
    "paper_id": "160",
    "table_id": "1",
    "panel_identifier": "2",
    "model_type": "log-linear",
    "comments": (
        "Table 1, Column (2): ln(cost) on 10-km distance bins (0–10 omitted) + "
        "weather/topography quadratics + south + fuelmodel dummies; FE: unit, "
        "state×mofy, state×year."
    ),
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest="distancebin_(10, 20]",
    fe=fe_idx,              # absorb unit & state-time FE
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
