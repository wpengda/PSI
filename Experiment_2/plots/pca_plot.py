import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

# ===== CONFIGURATION =====
INPUT_DIR = "All_Data_json"
HUMAN_FILE = os.path.join(INPUT_DIR, "human_normal.json")
OUTPUT_DIR = "plots"
OUTFILE = os.path.join(OUTPUT_DIR, "pca_all_levels.pdf") 

# Three LLM panel source files
PSI_FILE     = os.path.join(INPUT_DIR, "psi_GPT4o_zero_shot.json")
PERSONA_FILE = os.path.join(INPUT_DIR, "persona_GPT4o_zero_shot.json")
SHAPE_FILE   = os.path.join(INPUT_DIR, "shape_GPT4o_zero_shot.json")

# 5 traits
TRAITS = ["Extraversion","Agreeableness","Conscientiousness","Neuroticism","Openness"]

# 15 facets
FACETS = [
    "Sociability","Assertiveness","Energy_Level",
    "Compassion","Respectfulness","Trust",
    "Organization","Productiveness","Responsibility",
    "Anxiety","Depression","Emotional_Volatility",
    "Intellectual_Curiosity","Aesthetic_Sensitivity","Creative_Imagination"
]

# 60 items
ITEMS = [f"Item{i}" for i in range(1, 61)]

# Colors
COLOR_HUMAN = "#1f77b4"
COLOR_LLM   = "#ff7f0e"

# ===== FONT SIZE CONTROL =====
FONT_SIZE = 30  # <<< Change this to adjust all font sizes globally


# ========== UTILITY FUNCTIONS ==========
def iter_records(obj):
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it
    elif isinstance(obj, dict):
        if any(isinstance(v,(int,float,str)) for v in obj.values()):
            yield obj
        for v in obj.values():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        yield it
            elif isinstance(v, dict):
                yield v

def load_vectors(fp: str, fields) -> np.ndarray:
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rows = []
    for rec in iter_records(obj):
        vec, ok = [], True
        for t in fields:
            if t not in rec:
                ok = False; break
            try:
                v = float(rec[t])
            except (TypeError, ValueError):
                ok = False; break
            if not (1.0 <= v <= 5.0):
                ok = False; break
            vec.append(v)
        if ok:
            rows.append(vec)
    return np.array(rows, dtype=float) if rows else np.empty((0, len(fields)))

def fit_pca_on_human(X_h: np.ndarray) -> PCA:
    if X_h.size == 0:
        raise RuntimeError("Human data is empty or invalid.")
    pca = PCA(n_components=2, random_state=0)
    pca.fit(X_h)
    for k in range(2):
        if pca.components_[k].sum() < 0:
            pca.components_[k] *= -1
    return pca

def project_with_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    return pca.transform(X) if X.size else np.empty((0,2))

def draw_panel(ax, Zh: np.ndarray, Zl: np.ndarray, label: str):
    ax.scatter(Zh[:,0], Zh[:,1], s=12, alpha=0.6, c=COLOR_HUMAN)
    ax.scatter(Zl[:,0], Zl[:,1], s=12, alpha=0.6, c=COLOR_LLM)
    ax.set_xlabel("PC1", fontsize=FONT_SIZE)
    ax.set_ylabel("PC2", fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=FONT_SIZE-1)
    ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_SIZE, fontweight="bold")

def get_projected(fields):
    X_h   = load_vectors(HUMAN_FILE, fields)
    X_psi = load_vectors(PSI_FILE, fields)
    X_per = load_vectors(PERSONA_FILE, fields)
    X_shp = load_vectors(SHAPE_FILE, fields)

    pca = fit_pca_on_human(X_h)
    Zh  = project_with_pca(pca, X_h)
    Zpsi = project_with_pca(pca, X_psi)
    Zper = project_with_pca(pca, X_per)
    Zshp = project_with_pca(pca, X_shp)
    return Zh, Zpsi, Zper, Zshp


# ========== MAIN PIPELINE ==========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fp in [HUMAN_FILE, PSI_FILE, PERSONA_FILE, SHAPE_FILE]:
        if not os.path.isfile(fp):
            raise FileNotFoundError(os.path.abspath(fp))

    datasets = [
        (ITEMS,   "Items"),
        (FACETS,  "Facets"),
        (TRAITS,  "Domains"),
    ]

    # Plot: 3 rows × 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for row, (fields, row_label) in enumerate(datasets):
        Zh, Zpsi, Zper, Zshp = get_projected(fields)
        draw_panel(axes[row,0], Zh, Zpsi, "PSI")
        draw_panel(axes[row,1], Zh, Zper, "Persona")
        draw_panel(axes[row,2], Zh, Zshp, "Shape")

        # Row label
        axes[row,0].set_ylabel(row_label, fontsize=FONT_SIZE+1, fontweight="bold", labelpad=20)

    # Unified legend at the top
    handles = [
        Line2D([0],[0], marker="o", linestyle="none", color=COLOR_HUMAN, label="Human"),
        Line2D([0],[0], marker="o", linestyle="none", color=COLOR_LLM,   label="LLM"),
    ]
    fig.legend(handles=handles, labels=["Human","LLM"],
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=FONT_SIZE)

    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(OUTFILE, dpi=150, format="pdf")
    plt.close()
    print("Output figure:", os.path.abspath(OUTFILE))


if __name__ == "__main__":
    main()





