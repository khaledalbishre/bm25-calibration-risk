"""
Publication figures for the manuscript (results/figures_final/).

Style follows the principles of github.com/AndreyChurkin/BeautifulFigures:
vector output, figure size equal to the printed size so that fonts are
readable at their final size, few harmonious colours used with a fixed
meaning, light grid below the data, frameless legends, no decoration.

No new analysis is run. Every plotted value is either read from an
existing result table or recomputed with the exact code path that
produced that table, and the recomputation is checked against the table
before anything is drawn (the build stops on any mismatch).

    Figure 1  fig1_reliability_dl20       code path of 04_fit_calibrators.py;
                                          checked against tab_calibration_twofold.csv
    Figure 2  fig2_risk_coverage          code path of 07_fold_aware_risk.py;
                                          checked against tab_risk_foldaware.csv
    Figure 3  fig3_calibration_vs_risk    tab_calibration_pooled_oof.csv + tab_risk_foldaware.csv
    Figure 4  fig4_depth_matched          tab_risk_foldaware.csv + tab_risk_contrasts.csv

Usage:
    python code/08_final_figures.py
Writes:
    results/figures_final/fig{1..4}_*.pdf / .png
    results/figures_final/figure_values.csv   (every plotted value + its check)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

common = import_module("00_common")
m04 = import_module("04_fit_calibrators")
m07 = import_module("07_fold_aware_risk")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

OUT = common.RESULTS_DIR / "figures_final"
OUT.mkdir(parents=True, exist_ok=True)
TOL = 1e-9

# ---------------------------------------------------------------------
# Shared style. Figures are drawn at their printed width (8.9 cm) so the
# 8 pt labels and 7.5 pt tick labels are the sizes the reader sees.
# ---------------------------------------------------------------------
WIDTH_IN = 3.5
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#4e4e4e",
    "axes.labelcolor": "#262626",
    "xtick.color": "#4e4e4e",
    "ytick.color": "#4e4e4e",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "pdf.fonttype": 42,   # embed TrueType fonts (editable, journal-safe)
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

# One meaning per colour across all figures.
GREY, GREY_D = "#9a9a9a", "#4e4e4e"        # raw score / norm (no fitted calibration)
PURPLE, PURPLE_D = "#9671bd", "#6a408d"    # Platt
OCHRE, OCHRE_D = "#d9a441", "#a0731c"      # isotonic
TEAL, TEAL_D = "#77b5b6", "#378d94"        # score dispersion (CV@100 baseline)
REF = "#8a8a8a"                            # reference lines

values = []  # rows of figure_values.csv


def record(fig, item, value, source, check, ok):
    if not ok:
        raise SystemExit(f"VALUE CHECK FAILED: {fig} / {item}: {check} -> {value}")
    values.append({"figure": fig, "item": item, "value": value, "source": source, "check": check})


def style_axes(ax, grid_x=True, grid_y=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_y:
        ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    if grid_x:
        ax.grid(True, axis="x", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote results/figures_final/{name}.{{pdf,png}}")


# ---------------------------------------------------------------------
# Figure 1: reliability diagram (DL20), code path of 04_fit_calibrators
# ---------------------------------------------------------------------
def figure1():
    train = pd.read_parquet(common.DATA_DIR / "calib_frame_dl19.parquet")
    test = pd.read_parquet(common.DATA_DIR / "calib_frame_dl20.parquet").copy()
    platt = LogisticRegression(penalty=None, max_iter=10000)
    platt.fit(train[["bm25_score"]].to_numpy(), train["label"].to_numpy())
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train["bm25_score"].to_numpy(), train["label"].to_numpy())
    # Authoritative normalization: per-query min-max over the full top-1000 run.
    test["p_norm"] = m07.norm_of_frame(test, m07.load_run("dl20"))
    test["p_platt"] = platt.predict_proba(test[["bm25_score"]].to_numpy())[:, 1]
    test["p_isotonic"] = iso.predict(test["bm25_score"].to_numpy())

    tab = pd.read_csv(common.RESULTS_DIR / "tab_calibration_twofold.csv")
    tab = tab[tab.direction == "dl19->dl20"].set_index("estimator")
    labels = test["label"].to_numpy()
    series = []
    for name, col in [("norm", "p_norm"), ("platt", "p_platt"), ("isotonic", "p_isotonic")]:
        p = test[col].to_numpy()
        # Same check as the table: ECE@10 and Brier from these predictions.
        e10, b = m04.ece(p, labels, 10), m04.brier(p, labels)
        record("Fig1", f"{name} ECE@10", e10, "tab_calibration_twofold.csv (dl19->dl20)",
               f"recomputed {e10:.6f} vs table {tab.loc[name, 'ece_10bin_pooled']:.6f}",
               abs(e10 - tab.loc[name, "ece_10bin_pooled"]) < TOL)
        record("Fig1", f"{name} Brier", b, "tab_calibration_twofold.csv (dl19->dl20)",
               f"recomputed {b:.6f} vs table {tab.loc[name, 'brier']:.6f}",
               abs(b - tab.loc[name, "brier"]) < TOL)
        # Bin points exactly as in 04_fit_calibrators.py (10 equal-width bins).
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, 9)
        xs, ys, ns = [], [], []
        for k in range(10):
            m = idx == k
            if m.any():
                xs.append(p[m].mean()); ys.append(labels[m].mean()); ns.append(int(m.sum()))
        # Bin-weighted |acc - conf| must reproduce the same ECE@10.
        e_bins = sum(n / len(p) * abs(yv - xv) for xv, yv, n in zip(xs, ys, ns))
        record("Fig1", f"{name} bins reproduce ECE@10", e_bins, "tab_calibration_twofold.csv (dl19->dl20)",
               f"sum_b n_b/N|acc-conf| = {e_bins:.6f}", abs(e_bins - tab.loc[name, "ece_10bin_pooled"]) < TOL)
        for k, (xv, yv, n) in enumerate(zip(xs, ys, ns)):
            values.append({"figure": "Fig1", "item": f"{name} bin{k} (conf, acc, n)",
                           "value": f"{xv:.4f}, {yv:.4f}, {n}", "source": "calib_frame_dl19/dl20.parquet",
                           "check": "bin of the ECE check above"})
        series.append((name, xs, ys))

    style = {"norm": (GREY, GREY_D, "o", "-", "norm (min–max)"),
             "platt": (PURPLE, PURPLE_D, "s", "-", "Platt"),
             "isotonic": (OCHRE, OCHRE_D, "D", "-", "isotonic")}
    fig, ax = plt.subplots(figsize=(WIDTH_IN, WIDTH_IN * 0.93))
    ax.plot([0, 1], [0, 1], color=REF, linestyle="--", linewidth=0.9, zorder=1,
            label="perfect calibration")
    for name, xs, ys in series:
        fc, ec, mk, ls, lab = style[name]
        ax.plot(xs, ys, color=ec, linestyle=ls, linewidth=1.2, marker=mk,
                markerfacecolor=fc, markeredgecolor=ec, markeredgewidth=0.8,
                markersize=4.5 if mk != "D" else 4.0, zorder=3, label=lab)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Mean predicted probability of relevance (bin)")
    ax.set_ylabel("Observed relevance frequency (bin)")
    style_axes(ax)
    ax.legend(loc="upper left", borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig1_reliability_dl20")


# ---------------------------------------------------------------------
# Figure 2: fold-aware risk-coverage, code path of 07_fold_aware_risk
# ---------------------------------------------------------------------
def figure2():
    runs = {s: m07.load_run(s) for s in m07.FOLDS}
    frames = {s: pd.read_parquet(common.DATA_DIR / f"calib_frame_{s}.parquet") for s in m07.FOLDS}
    ndcg10 = {s: m07.m05.per_query_ndcg10(s) for s in m07.FOLDS}
    parts = []
    for train_split, predict_split in (("dl19", "dl20"), ("dl20", "dl19")):
        df = m07.fold_signals(train_split, predict_split, runs, frames)
        df["ndcg10"] = pd.Series(ndcg10[predict_split])
        df["risk"] = (df["ndcg10"] < df["ndcg10"].median()).astype(int)
        parts.append(df)
    P = pd.concat(parts)
    folds, y = P["fold"].to_numpy(), P["risk"].to_numpy()

    tab = pd.read_csv(common.RESULTS_DIR / "tab_risk_foldaware.csv").set_index("signal")
    shown = [("cv100", "CV@100 (NQC-style baseline)", TEAL_D, "-"),
             ("qrcmean_iso", "QRC-mean (isotonic)", OCHRE_D, "-"),
             ("top1", "Raw top-1 score", GREY_D, (0, (4, 1.5)))]
    curves = []
    for col, lab, color, ls in shown:
        s = P[col].to_numpy(dtype=float)
        for f in m07.FOLDS:
            m = folds == f
            a = m07.risk_auc(y[m], s[m])
            record("Fig2", f"{col} AUROC {f}", a, "tab_risk_foldaware.csv",
                   f"recomputed {a:.6f} vs table {tab.loc[col, f'auroc_{f}']:.6f}",
                   abs(a - tab.loc[col, f"auroc_{f}"]) < TOL)
        a = m07.stratified_auc(y, s, folds)
        record("Fig2", f"{col} stratified AUROC", a, "tab_risk_foldaware.csv",
               f"recomputed {a:.6f} vs table {tab.loc[col, 'auroc_stratified']:.6f}",
               abs(a - tab.loc[col, "auroc_stratified"]) < TOL)
        order = np.argsort(-m07.within_fold_rank(s, folds), kind="stable")
        bad = np.cumsum(y[order])
        n = np.arange(1, len(order) + 1)
        curves.append((n / len(order), bad / n, lab, color, ls))
        values.append({"figure": "Fig2", "item": f"{col} risk at coverage 0.25/0.50/0.75/1.00",
                       "value": ", ".join(f"{(bad / n)[int(round(c * len(n))) - 1]:.4f}" for c in (0.25, 0.5, 0.75, 1.0)),
                       "source": "runs/*.trec + calib frames (07 code path)", "check": "curve of the AUROC check above"})
    base = y.mean()
    record("Fig2", "no-selection rate", base, "tab_fold_summary.csv",
           f"{y.sum()}/{len(y)} = {base:.4f}; table n_at_risk 21+27 of 43+54", y.sum() == 48 and len(y) == 97)

    fig, ax = plt.subplots(figsize=(WIDTH_IN, WIDTH_IN * 0.78))
    ax.axhline(base, color=REF, linestyle=":", linewidth=1.0, zorder=1)
    ax.text(0.015, base + 0.012, f"no selection ({base:.2f})", color=GREY_D, fontsize=7, va="bottom")
    for x, r, lab, color, ls in curves:
        ax.plot(x, r, color=color, linestyle=ls, linewidth=1.3, label=lab, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Coverage (fraction of queries retained, most confident first)")
    ax.set_ylabel("Below-median rate among retained queries")
    style_axes(ax)
    ax.legend(loc="lower right", borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig2_risk_coverage")


# ---------------------------------------------------------------------
# Figure 3: calibration quality vs query-level risk utility (tables only)
# ---------------------------------------------------------------------
def figure3():
    cal = pd.read_csv(common.RESULTS_DIR / "tab_calibration_pooled_oof.csv").set_index("estimator")
    risk = pd.read_csv(common.RESULTS_DIR / "tab_risk_foldaware.csv").set_index("signal")
    # estimator -> (QRC-mean signal, label, face, edge, marker, filled)
    est = [("norm", "qrcmean_norm", "norm (no fitting)", GREY, GREY_D, "o", True),
           ("platt", "qrcmean_platt", "Platt", PURPLE, PURPLE_D, "s", True),
           ("isotonic", "qrcmean_iso", "isotonic", OCHRE, OCHRE_D, "D", True),
           ("norm+platt", "qrcmean_nplatt", "norm+Platt", PURPLE, PURPLE_D, "s", False),
           ("norm+isotonic", "qrcmean_niso", "norm+isotonic", OCHRE, OCHRE_D, "D", False)]
    # Manuscript Tables 3 and 4 (3 decimals) must agree with the CSVs.
    printed = {"norm": (0.219, 0.562), "platt": (0.211, 0.608), "isotonic": (0.211, 0.599),
               "norm+platt": (0.201, 0.562), "norm+isotonic": (0.202, 0.568)}
    pts = []
    for e, sig, lab, fc, ec, mk, filled in est:
        b, a = cal.loc[e, "brier_pooled_oof"], risk.loc[sig, "auroc_stratified"]
        lo, hi = risk.loc[sig, "auroc_ci_lo"], risk.loc[sig, "auroc_ci_hi"]
        ok = round(b, 3) == printed[e][0] and round(a, 3) == printed[e][1]
        record("Fig3", f"{lab}: Brier, stratified AUROC [CI]", f"{b:.4f}, {a:.4f} [{lo:.4f}, {hi:.4f}]",
               "tab_calibration_pooled_oof.csv; tab_risk_foldaware.csv",
               f"rounds to Table 3/4 values {printed[e]}", ok)
        pts.append((b, a, lo, hi, lab, fc, ec, mk, filled))
    cv = risk.loc["cv100"]
    record("Fig3", "CV@100 reference AUROC [CI]", f"{cv.auroc_stratified:.4f} [{cv.auroc_ci_lo:.4f}, {cv.auroc_ci_hi:.4f}]",
           "tab_risk_foldaware.csv", "rounds to 0.731 [0.625, 0.824]",
           (round(cv.auroc_stratified, 3), round(cv.auroc_ci_lo, 3), round(cv.auroc_ci_hi, 3)) == (0.731, 0.625, 0.824))

    # Direct labels, anchored in data coordinates as (x, y, ha, leader line).
    # Positions keep every label clear of other labels, the confidence
    # intervals, the markers, the CV@100 line and band, the chance line and
    # the legend; check_layout() verifies this on the rendered figure.
    label_pos = {
        "norm+Platt": (0.2031, 0.519, "left", True),
        "norm+isotonic": (0.2031, 0.560, "left", False),
        "Platt": (0.2121, 0.603, "left", False),
        "isotonic": (0.2103, 0.604, "right", False),
        "norm (no fitting)": (0.2181, 0.553, "right", False),
    }
    fig, ax = plt.subplots(figsize=(WIDTH_IN, WIDTH_IN * 0.92))
    band = ax.axhspan(cv.auroc_ci_lo, cv.auroc_ci_hi, color=TEAL, alpha=0.13, linewidth=0, zorder=0)
    cv_line = ax.axhline(cv.auroc_stratified, color=TEAL_D, linestyle="--", linewidth=1.0, zorder=1)
    cv_txt = ax.text(0.2212, cv.auroc_stratified + 0.008, "CV@100 baseline and 95% CI (no calibration)",
                     color=TEAL_D, fontsize=7, ha="right", va="bottom")
    chance_line = ax.axhline(0.5, color=REF, linestyle=":", linewidth=0.9, zorder=1)
    chance_txt = ax.text(0.2150, 0.503, "chance", color=GREY_D, fontsize=7, ha="center", va="bottom")
    ci_lines, markers, labels = [], [], []
    for b, a, lo, hi, lab, fc, ec, mk, filled in pts:
        ci_lines += ax.plot([b, b], [lo, hi], color=ec, linewidth=0.8, alpha=0.55, zorder=2,
                            solid_capstyle="butt")
        markers += ax.plot(b, a, marker=mk, markersize=6 if mk != "D" else 5.4, markeredgewidth=1.1,
                           markerfacecolor=fc if filled else "white", markeredgecolor=ec,
                           linestyle="none", zorder=4)
        x, y, ha, leader = label_pos[lab]
        labels.append(ax.text(x, y, f"{lab}\n{a:.3f}", ha=ha, va="center", fontsize=7, color=ec,
                              linespacing=1.0, zorder=5))
        if leader:
            ax.annotate("", xy=(b, a), xytext=(x, y), zorder=3,
                        arrowprops=dict(arrowstyle="-", color=ec, linewidth=0.5, alpha=0.8,
                                        shrinkA=1.5, shrinkB=4.0))
    ax.set_xlim(0.1975, 0.2215)
    ax.set_ylim(0.42, 0.85)
    ax.set_xticks([0.20, 0.21, 0.22])
    ax.set_xlabel("Pooled out-of-fold Brier score  (← better calibration)")
    ax.set_ylabel("Stratified AUROC of QRC-mean  (better →)")
    style_axes(ax)
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="#d0d0d0",
                      markeredgecolor=GREY_D, markersize=5, label="Platt / isotonic fitted on raw BM25 score"),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                      markeredgecolor=GREY_D, markersize=5, label="Platt / isotonic fitted on min–max normalized score")]
    # Legend above the axes, so it does not sit on the CV@100 band.
    legend = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2,
                       handletextpad=0.3)
    fig.tight_layout(pad=0.3)

    def check_layout():
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        box = lambda art, pad=1.0: art.get_window_extent(r).expanded(1.0, 1.0).padded(pad)
        inside = ax.get_window_extent(r)
        named = dict(zip([t.get_text().split("\n")[0] for t in labels], labels))
        texts = {**named, "CV@100 label": cv_txt, "chance label": chance_txt}
        obstacles = ({f"CI {k}": l for k, l in zip(named, ci_lines)}
                     | {f"marker {k}": m for k, m in zip(named, markers)}
                     | {"CV@100 line": cv_line, "chance line": chance_line, "legend": legend})
        problems = []
        for name, t in texts.items():
            tb = box(t)
            if not (inside.x0 <= tb.x0 and tb.x1 <= inside.x1 and inside.y0 <= tb.y0 and tb.y1 <= inside.y1):
                problems.append(f"{name} leaves the axes")
            others = {k: v for k, v in texts.items() if k != name} | obstacles
            if name in named:
                others["CV@100 band"] = band
            for oname, o in others.items():
                if oname in (f"CI {name}", f"marker {name}") and name not in named:
                    continue
                if name == "CV@100 label" and oname == "CV@100 line":
                    continue
                if name == "chance label" and oname == "chance line":
                    continue
                if tb.overlaps(box(o, 0.5)):
                    problems.append(f"{name} touches {oname}")
        return problems

    problems = check_layout()
    record("Fig3", "direct-label layout", "; ".join(problems) or "no collisions",
           "rendered figure", "labels clear of labels, CIs, markers, band, chance line, legend, axes",
           not problems)
    save(fig, "fig3_calibration_vs_risk")


# ---------------------------------------------------------------------
# Figure 4: depth-matched diagnostic (tables only)
# ---------------------------------------------------------------------
def figure4():
    risk = pd.read_csv(common.RESULTS_DIR / "tab_risk_foldaware.csv").set_index("signal")
    con = pd.read_csv(common.RESULTS_DIR / "tab_risk_contrasts.csv")
    con = con.set_index(con["signal_a"] + "-" + con["signal_b"])
    # Manuscript Table 5 values (stratified AUROC [95% CI]).
    printed = {"mean10": (0.610, 0.489, 0.720), "std10": (0.647, 0.534, 0.754), "cv10": (0.589, 0.468, 0.695),
               "mean100": (0.533, 0.409, 0.651), "std100": (0.769, 0.668, 0.856), "cv100": (0.731, 0.625, 0.824)}
    for s, exp in printed.items():
        r = risk.loc[s]
        got = (round(r.auroc_stratified, 3), round(r.auroc_ci_lo, 3), round(r.auroc_ci_hi, 3))
        record("Fig4", f"{s} AUROC [CI]", f"{r.auroc_stratified:.4f} [{r.auroc_ci_lo:.4f}, {r.auroc_ci_hi:.4f}]",
               "tab_risk_foldaware.csv", f"rounds to Table 5 {exp}", got == exp)
    # Manuscript Table 6 contrasts annotated on the figure.
    printed_c = {"std10-mean10": (0.038, -0.098, 0.176), "std100-mean100": (0.236, 0.097, 0.369)}
    for k, exp in printed_c.items():
        r = con.loc[k]
        got = (round(r.delta_auroc, 3), round(r.ci_lo, 3), round(r.ci_hi, 3))
        record("Fig4", f"contrast {k}", f"{r.delta_auroc:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]",
               "tab_risk_contrasts.csv", f"rounds to Table 6 {exp}", got == exp)

    stats = [("mean", "Level: mean of scores", GREY, GREY_D, "o", True, -0.13),
             ("std", "Dispersion: std of scores", TEAL, TEAL_D, "o", True, 0.0),
             ("cv", "Dispersion: CV of scores", TEAL, TEAL_D, "o", False, 0.13)]
    xpos = {10: 0.0, 100: 1.0}
    fig, ax = plt.subplots(figsize=(WIDTH_IN, WIDTH_IN * 0.92))
    ax.axhline(0.5, color=REF, linestyle=":", linewidth=0.9, zorder=1)
    ax.text(1.42, 0.503, "chance", color=GREY_D, fontsize=7, ha="right", va="bottom")
    for key, lab, fc, ec, mk, filled, off in stats:
        xs, ys = [], []
        for depth in (10, 100):
            r = risk.loc[f"{key}{depth}"]
            x = xpos[depth] + off
            ax.plot([x, x], [r.auroc_ci_lo, r.auroc_ci_hi], color=ec, linewidth=0.9, alpha=0.6,
                    zorder=2, solid_capstyle="butt")
            xs.append(x); ys.append(r.auroc_stratified)
        ax.plot(xs, ys, color=ec, linewidth=1.0, alpha=0.7,
                linestyle="-" if filled else (0, (3, 1.5)), zorder=3)
        ax.plot(xs, ys, linestyle="none", marker=mk, markersize=6, markeredgewidth=1.1,
                markerfacecolor=fc if filled else "white", markeredgecolor=ec, zorder=4, label=lab)
    for depth, k, y_txt in ((10, "std10-mean10", 0.985), (100, "std100-mean100", 0.985)):
        r = con.loc[k]
        sig = r.ci_lo > 0 or r.ci_hi < 0
        txt = (f"std − mean: {r.delta_auroc:+.3f}\n[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
               .replace("-", "−") + ("\nCI excludes 0" if sig else "\nCI includes 0"))
        ax.text(xpos[depth], y_txt, txt, ha="center", va="top", fontsize=7,
                color=TEAL_D if sig else GREY_D, linespacing=1.15)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0.38, 0.99)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Top 10 scores", "Top 100 scores"])
    ax.set_xlabel("Depth of the ranked list summarized")
    ax.set_ylabel("Stratified AUROC (95% CI)")
    style_axes(ax, grid_x=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, borderaxespad=0.2,
              handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    save(fig, "fig4_depth_matched")


def main():
    figure1()
    figure2()
    figure3()
    figure4()
    pd.DataFrame(values).to_csv(OUT / "figure_values.csv", index=False)
    n_checks = sum(1 for v in values if not str(v["check"]).endswith("above"))
    print(f"figure_values.csv: {len(values)} rows, {n_checks} value checks passed")


if __name__ == "__main__":
    main()
