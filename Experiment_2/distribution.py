# -*- coding: utf-8 -*-
"""
Compute per-dimension sample standard deviations (SD) for human vs model,
then correlate SD vectors to assess whether the model reproduces the
human variability pattern.

Granularities:
  - Items (60 dims: Item1..Item60)
  - Facets (15 dims)
  - Domains (5 dims)

Metrics reported (per granularity):
  - Pearson_r, Pearson_p
  - Spearman_rho, Spearman_p
  - RMSE between SD vectors
  - MAE between SD vectors

Output CSV is strictly ordered by the 21 model+setting labels you specified.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "All_Data_json"

ITEM_KEYS = [f"Item{i}" for i in range(1, 61)]
FACET_KEYS = [
    "Sociability","Assertiveness","Energy_Level",
    "Compassion","Respectfulness","Trust",
    "Organization","Productiveness","Responsibility",
    "Anxiety","Depression","Emotional_Volatility",
    "Intellectual_Curiosity","Aesthetic_Sensitivity","Creative_Imagination"
]
DOMAIN_KEYS = ["Extraversion","Agreeableness","Conscientiousness","Neuroticism","Openness"]

# Desired output order
DESIRED_ORDER = [
    "Mistral-7B-Instruct+Persona","Mistral-7B-Instruct+Shape","Mistral-7B-Instruct+PSI",
    "Gemma-2-9B-IT+Persona","Gemma-2-9B-IT+Shape","Gemma-2-9B-IT+PSI",
    "Gemma-2-27B-IT+Persona","Gemma-2-27B-IT+Shape","Gemma-2-27B-IT+PSI",
    "Llama3-8B-Instruct+Persona","Llama3-8B-Instruct+Shape","Llama3-8B-Instruct+PSI",
    "Llama3-70B-Instruct+Persona","Llama3-70B-Instruct+Shape","Llama3-70B-Instruct+PSI",
    "GPT4o-mini+Persona","GPT4o-mini+Shape","GPT4o-mini+PSI",
    "GPT4o+Persona","GPT4o+Shape","GPT4o+PSI"
]

# ----------------------------
# IO Helpers
# ----------------------------
def load_json_as_dataframe(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        df = pd.json_normalize(data)
    elif isinstance(data, dict):
        # dict[str, list] or dict[str, scalar]
        if all(isinstance(v, list) for v in data.values()):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported JSON format: {path}")
    # best-effort numeric coercion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def intersect_columns(df_model: pd.DataFrame, df_human: pd.DataFrame, keys):
    common = [k for k in keys if k in df_model.columns and k in df_human.columns]
    return common

def sd_vector(df: pd.DataFrame, keys):
    """Compute sample SD (ddof=1) per column over available rows; drop NaN rows per column independently."""
    sds = []
    used = []
    for k in keys:
        col = pd.to_numeric(df[k], errors="coerce") if k in df.columns else None
        if col is None:
            continue
        col = col.dropna()
        if col.shape[0] >= 2:
            sds.append(float(np.std(col.values.astype(float), ddof=1)))
            used.append(k)
        else:
            # not enough samples -> skip this dim
            pass
    return np.array(sds, dtype=float), used

# ----------------------------
# Correlation between SD patterns
# ----------------------------
def correlate_sd_vectors(sd_model: np.ndarray, sd_human: np.ndarray, dims: list):
    """
    Pair align by order in 'dims'. sd_model and sd_human must correspond to dims in the same order.
    Returns Pearson r/p, Spearman rho/p, RMSE, MAE.
    """
    # ensure same length
    if sd_model.shape != sd_human.shape or sd_model.size == 0:
        return dict(Pearson_r=np.nan, Pearson_p=np.nan,
                    Spearman_rho=np.nan, Spearman_p=np.nan,
                    RMSE=np.nan, MAE=np.nan, dims_used=len(dims))
    try:
        r, p = pearsonr(sd_model, sd_human)
    except Exception:
        r, p = (np.nan, np.nan)
    try:
        rho, p_s = spearmanr(sd_model, sd_human)
    except Exception:
        rho, p_s = (np.nan, np.nan)
    diff = sd_model - sd_human
    rmse = float(np.sqrt(np.mean(diff**2))) if diff.size > 0 else np.nan
    mae = float(np.mean(np.abs(diff))) if diff.size > 0 else np.nan
    return dict(Pearson_r=float(r), Pearson_p=float(p),
                Spearman_rho=float(rho), Spearman_p=float(p_s),
                RMSE=rmse, MAE=mae, dims_used=len(dims))

def build_aligned_sd_vectors(df_model, df_human, keys):
    """
    1) Find common keys.
    2) Compute SD per key for model and human independently.
    3) Keep intersection of keys that BOTH have valid SDs (>=2 samples).
    4) Return aligned vectors in the same order.
    """
    common = intersect_columns(df_model, df_human, keys)
    # compute per side SD for common
    sd_m_map = {}
    sd_h_map = {}
    for k in common:
        col_m = pd.to_numeric(df_model[k], errors="coerce") if k in df_model.columns else None
        col_h = pd.to_numeric(df_human[k], errors="coerce") if k in df_human.columns else None

        if col_m is not None:
            v = col_m.dropna().values.astype(float)
            if v.shape[0] >= 2:
                sd_m_map[k] = float(np.std(v, ddof=1))
        if col_h is not None:
            v = col_h.dropna().values.astype(float)
            if v.shape[0] >= 2:
                sd_h_map[k] = float(np.std(v, ddof=1))

    # intersection of keys that have valid SD on both sides
    dims = [k for k in common if (k in sd_m_map and k in sd_h_map)]
    sd_m = np.array([sd_m_map[k] for k in dims], dtype=float)
    sd_h = np.array([sd_h_map[k] for k in dims], dtype=float)
    return sd_m, sd_h, dims

# ----------------------------
# File label mapping
# ----------------------------
def normalize_name(s: str) -> str:
    s = s.lower().replace("-", "_").replace(" ", "_")
    return s

def map_filename_to_label(fname: str) -> str:
    f = normalize_name(fname)
    if   "persona" in f: tag = "+Persona"
    elif "shape"   in f: tag = "+Shape"
    elif "psi"     in f: tag = "+PSI"
    else:                tag = ""

    if "mistral" in f and "7b" in f:
        base = "Mistral-7B-Instruct"
    elif "gemma2" in f and "9b" in f:
        base = "Gemma-2-9B-IT"
    elif "gemma2" in f and "27b" in f:
        base = "Gemma-2-27B-IT"
    elif "llama3" in f and "8b" in f:
        base = "Llama3-8B-Instruct"
    elif "llama3" in f and "70b" in f:
        base = "Llama3-70B-Instruct"
    elif ("gpt4o_mini" in f) or ("gpt4o-mini" in f) or ("gpt4omini" in f):
        base = "GPT4o-mini"
    elif "gpt4o" in f:
        base = "GPT4o"
    else:
        base = fname
    return base + tag

# ----------------------------
# Main
# ----------------------------
def main():
    human_path = os.path.join(DATA_DIR, "human_normal.json")
    if not os.path.isfile(human_path):
        raise SystemExit(f"Missing {human_path}")
    df_human = load_json_as_dataframe(human_path)

    model_files = []
    for pat in ["persona_*.json", "psi_*.json", "shape_*.json"]:
        model_files.extend(glob.glob(os.path.join(DATA_DIR, pat)))
    if not model_files:
        raise SystemExit("No model files found under All_Data_json/")

    rows = []
    for fpath in sorted(model_files):
        name = os.path.basename(fpath)
        label = map_filename_to_label(name)
        df_model = load_json_as_dataframe(fpath)

        # Items
        sd_m, sd_h, dims = build_aligned_sd_vectors(df_model, df_human, ITEM_KEYS)
        items = correlate_sd_vectors(sd_m, sd_h, dims)

        # Facets
        sd_m, sd_h, dims = build_aligned_sd_vectors(df_model, df_human, FACET_KEYS)
        facets = correlate_sd_vectors(sd_m, sd_h, dims)

        # Domains
        sd_m, sd_h, dims = build_aligned_sd_vectors(df_model, df_human, DOMAIN_KEYS)
        domains = correlate_sd_vectors(sd_m, sd_h, dims)

        rows.append({
            "model_file": name,
            "label": label,

            "Items_SD_Pearson_r": items["Pearson_r"],
            "Items_SD_Pearson_p": items["Pearson_p"],
            "Items_SD_Spearman_rho": items["Spearman_rho"],
            "Items_SD_Spearman_p": items["Spearman_p"],
            "Items_SD_RMSE": items["RMSE"],
            "Items_SD_MAE": items["MAE"],
            "Items_dims_used": items["dims_used"],

            "Facets_SD_Pearson_r": facets["Pearson_r"],
            "Facets_SD_Pearson_p": facets["Pearson_p"],
            "Facets_SD_Spearman_rho": facets["Spearman_rho"],
            "Facets_SD_Spearman_p": facets["Spearman_p"],
            "Facets_SD_RMSE": facets["RMSE"],
            "Facets_SD_MAE": facets["MAE"],
            "Facets_dims_used": facets["dims_used"],

            "Domains_SD_Pearson_r": domains["Pearson_r"],
            "Domains_SD_Pearson_p": domains["Pearson_p"],
            "Domains_SD_Spearman_rho": domains["Spearman_rho"],
            "Domains_SD_Spearman_p": domains["Spearman_p"],
            "Domains_SD_RMSE": domains["RMSE"],
            "Domains_SD_MAE": domains["MAE"],
            "Domains_dims_used": domains["dims_used"],
        })

    out = pd.DataFrame(rows)
    out["__order__"] = out["label"].apply(lambda x: DESIRED_ORDER.index(x) if x in DESIRED_ORDER else 999)
    out = out.sort_values(["__order__", "label"]).drop(columns="__order__")
    out.to_csv("results_sd_correlation.csv", index=False, encoding="utf-8")

    print("Done. Saved -> results_sd_correlation.csv")
    print("Columns: per-granularity SD correlations (Pearson/Spearman) + RMSE/MAE + dims_used")

if __name__ == "__main__":
    main()



