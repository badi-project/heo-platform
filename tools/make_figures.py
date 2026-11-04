#!/usr/bin/env python3
"""Regenerate the validation and screening figures from the pipeline outputs.

    figures/fig4_validation.png   grouped-CV confusion matrix, calibration, and
                                  accuracy under three validations vs baselines
    figures/fig5_candidats.png    predicted family shares across the screening,
                                  this version against v1.0 (equimolar cation sets)

Colours follow a validated categorical/sequential palette (blue first, orange second,
neutral grey for references). Requires matplotlib (requirements-train.txt).

    python tools/make_figures.py
"""
import csv
import gzip
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import features  # noqa: E402
import train_model as tm  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY, GREY_LIGHT = "#8b8a85", "#d9d8d3"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
       "#0d366b"]
FR = {"spinel": "spinelle", "perovskite": "pérovskite", "rocksalt": "rocksalt",
      "fluorite": "fluorine", "pyrochlore": "pyrochlore", "layered": "lamellaire"}

# v1.0 screening shares (data/screening_all.csv.gz at tag 1.0.0), kept here because the
# v1.0 files are no longer in the working tree
V1_IN_DOMAIN = {"perovskite": 65.9, "fluorite": 14.8, "pyrochlore": 10.7, "spinel": 6.2,
                "layered": 2.3, "rocksalt": 0.0}

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": GREY_LIGHT, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
    "axes.titlesize": 10.5, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def grouped_oof():
    rows = tm.load_rows()
    E = features.load_elements()
    b = json.load(open(os.path.join(ROOT, "app", "heo_model.json"), encoding="utf-8"))
    names, X = tm.featurize_rows(rows, E, b["onehot"])
    lab = np.array([r["proto"] in tm.CLASSES for r in rows])
    y = np.array([r["proto"] for r in rows])[lab]
    g = np.array([r["sysk"] for r in rows])[lab]
    pred, raw = tm.oof(tm.hgb, X[lab], y, g)
    return y, pred, raw, b


def fig_validation():
    y, pred, raw, b = grouped_oof()
    M, T = b["metrics"], b["model"]["temperature"]
    y_idx = np.array([tm.CLASSES.index(c) for c in y])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), gridspec_kw={"width_ratios": [1.05, 1, 1.25]})

    # (a) confusion matrix
    ax = axes[0]
    order = tm.CLASSES_ORDER
    cm = np.array([[np.sum((y == t) & (pred == p)) for p in order] for t in order])
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    for i in range(len(order)):
        for j in range(len(order)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8.5,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(order)), [FR[c] for c in order], rotation=40, ha="right")
    ax.set_yticks(range(len(order)), [FR[c] for c in order])
    ax.set_xlabel("famille prédite")
    ax.set_ylabel("famille publiée")
    ax.set_title(f"a  Validation groupée : exactitude {M['cv']['accuracy']:.1%}".replace(".", ","),
                 loc="left")
    for s in ax.spines.values():
        s.set_visible(False)

    # (b) reliability
    ax = axes[1]
    edges = np.linspace(0, 1, 11)
    for probs, col, lab, mk in ((tm.softmax(raw), GREY, "brutes", "o"),
                                (tm.softmax(raw / T), BLUE, f"calibrées (T = {T:.2f})", "o")):
        conf, hit = probs.max(axis=1), probs.argmax(axis=1) == y_idx
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean()); ys.append(hit[m].mean()); ns.append(m.sum())
        ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.5,
                label=lab.replace(".", ","))
    ax.plot([0, 1], [0, 1], color=GREY_LIGHT, lw=1, ls="--", zorder=0)
    ax.set_xlim(0.2, 1.01)
    ax.set_ylim(0.2, 1.01)
    ax.set_xlabel("probabilité annoncée (famille la plus probable)")
    ax.set_ylabel("exactitude observée")
    ax.grid(color=GREY_LIGHT, lw=0.5, alpha=0.6)
    ece_raw, ece_cal = M["cv"]["ece_raw"], M["cv"]["ece_cal"]
    ax.set_title(f"b  Calibration : ECE {ece_raw:.3f} → {ece_cal:.3f}".replace(".", ","),
                 loc="left")
    ax.legend(loc="lower right")

    # (c) accuracy under three validations vs references
    ax = axes[2]
    bl = M["baselines"]
    items = [
        ("validation groupée par système", M["cv"]["accuracy"], BLUE),
        (f"temporelle : articles {M['temporal']['cut'] + 1}+", M["temporal"]["accuracy"], BLUE),
        ("  dont systèmes jamais vus", M["temporal"]["accuracy_unseen"], BLUE),
        ("chimie inédite (groupes retirés)", M["leave_chemistry_out"]["accuracy"], BLUE),
        ("réf. : 9 descripteurs classiques", bl["classic_9"]["accuracy"], GREY),
        ("réf. : présence des cations", bl["cation_presence"]["accuracy"], GREY),
        ("réf. : système publié le plus proche", bl["nearest_system"]["accuracy"], GREY),
        ("réf. : classe majoritaire", bl["majority"]["accuracy"], GREY),
    ]
    ypos = np.arange(len(items))[::-1]
    for yp, (lab, v, col) in zip(ypos, items):
        ax.barh(yp, v, height=0.62, color=col, edgecolor=SURFACE, linewidth=2)
        ax.text(v + 0.012, yp, f"{v:.1%}".replace(".", ","), va="center", fontsize=8.5,
                color=INK)
    ax.set_yticks(ypos, [i[0] for i in items])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "25 %", "50 %", "75 %", "100 %"])
    ax.axvline(0, color=GREY_LIGHT, lw=1)
    ax.set_title("c  Exactitude : modèle (bleu) et références (gris)", loc="left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout(w_pad=2.2)
    out = os.path.join(ROOT, "figures", "fig4_validation.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


def fig_screening():
    c_in = Counter()
    n_in = 0
    with gzip.open(os.path.join(ROOT, "data", "screening_all.csv.gz"), "rt",
                   encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["in_domain"] == "1":
                n_in += 1
                c_in[r["proto"]] += 1
    order = tm.CLASSES_ORDER
    new = [100 * c_in[c] / n_in for c in order]
    old = [V1_IN_DOMAIN[c] for c in order]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, old, w, color=GREY, edgecolor=SURFACE, linewidth=2,
           label="v1.0 — ensembles de cations équimolaires")
    ax.bar(x + w / 2, new, w, color=BLUE, edgecolor=SURFACE, linewidth=2,
           label="v1.1 — stœchiométrie relue dans les formules")
    for xi, v in zip(x, new):
        ax.text(xi + w / 2, v + 1, f"{v:.0f} %", ha="center", fontsize=8.5, color=INK)
    for xi, v in zip(x, old):
        ax.text(xi - w / 2, v + 1, f"{v:.0f} %", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [FR[c] for c in order])
    ax.set_ylabel("part des compositions criblées\ndans le domaine d'applicabilité")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax.grid(axis="y", color=GREY_LIGHT, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Famille prédite sur le criblage équimolaire", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig5_candidats.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    fig_validation()
    fig_screening()
