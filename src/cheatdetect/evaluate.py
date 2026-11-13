"""Evaluation: compare the KB's verdicts with the real CS2CD labels.

Evaluation set = every player EXCEPT those in calibration matches (even-numbered clean
matches), so the thresholds never saw the negatives they are tested on.

Caveat for the report: "non-cheater" means "not labelled". The clean matches contain unlabelled
cheaters (see calibrate.py), so the measured precision is a LOWER bound.

Outputs:
  reports/figures/*.pdf|png   figures for Overleaf
  reports/tables/*.tex|csv    tables for Overleaf
  reports/metrics.json        headline numbers

Usage: python -m cheatdetect.evaluate
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cheatdetect import config  # noqa: E402
from cheatdetect.calibrate import is_calibration_match  # noqa: E402

FIG_DIR = config.REPORTS_DIR / "figures"
TAB_DIR = config.REPORTS_DIR / "tables"

# Validated default palette (dataviz skill, light mode, print).
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b5b4ae"
ACTION_ORDER = ["no_action", "clear", "monitor", "hide_from_leaderboard", "review_for_ban"]
ACTION_LABEL = {"no_action": "Insufficient evidence", "clear": "Clear", "monitor": "Monitor",
                "hide_from_leaderboard": "Hide from leaderboard", "review_for_ban": "Ban review (human)"}
ACTION_COLOR = {"no_action": GRAY, "clear": "#cde2fb", "monitor": "#86b6ef",
                "hide_from_leaderboard": "#2a78d6", "review_for_ban": "#104281"}
LEVELS = {  # operating point -> actions counted as "flagged"
    "monitor or stricter": {"monitor", "hide_from_leaderboard", "review_for_ban"},
    "hide or stricter": {"hide_from_leaderboard", "review_for_ban"},
    "ban review only": {"review_for_ban"},
}
RULE_TEXT = {
    "a1": "aimbot: headshot rate + accuracy", "a2": "aimbot: headshot rate + HS-kill rate",
    "a3": "aimbot: accuracy + high impact", "a4": "aimbot: headshot rate + high impact",
    "w1": "wallhack: wallbang rate", "w2": "wallhack: wallbang rate + high impact",
    "w3": "wallhack: smoke kills + high impact", "r1": "rage: airborne kills + headshot rate",
    "r2": "rage: blind kills + headshot rate", "r3": "rage: wallbang + headshot + accuracy",
    "l1": "legit outlier: high impact + normal aim", "c1": "cheating <- aimbot",
    "c2": "cheating <- wallhack", "c3": "cheating <- rage cheat", "c4": "cheating <- legit outlier (against)",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


def eval_set() -> pd.DataFrame:
    v = pd.read_parquet(config.VERDICTS_FILE)
    v = v[~v.match_id.map(is_calibration_match).astype(bool)].copy()
    v["clean_match"] = v.match_id.astype(str).str.startswith("n")
    return v


def pr_curve(y_true, score):
    """Precision and recall at every distinct score threshold (highest score first)."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]   # last index of each distinct score
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    return tp / (tp + fp), tp / y.sum()


def scores(y_true, y_pred) -> dict:
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum()); tn = int((~y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(flagged=tp + fp, tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=r,
                f1=2 * p * r / (p + r) if p + r else 0.0, fpr=fp / (fp + tn) if fp + tn else 0.0)


def operating_points(v: pd.DataFrame) -> pd.DataFrame:
    y = v.is_cheater.to_numpy()
    rows = [dict(level=name, **scores(y, v.action.isin(acts).to_numpy())) for name, acts in LEVELS.items()]
    # Baselines: single-signal rules with no knowledge base
    sym = v.symptoms.map(set)
    baselines = {
        "baseline: headshot rate > p95": sym.map(lambda s: bool({"abnormal(headshot_rate)@p95", "abnormal(headshot_rate)@p99"} & s)),
        "baseline: any metric > p99": sym.map(lambda s: any(x.endswith("@p99") for x in s)),
        "baseline: any metric > p95": sym.map(lambda s: any(x.startswith("abnormal(") for x in s)),
    }
    rows += [dict(level=name, **scores(y, pred.to_numpy())) for name, pred in baselines.items()]
    return pd.DataFrame(rows)


def fp_by_source(v: pd.DataFrame) -> pd.DataFrame:
    """Where do false positives come from: clean matches (possibly unlabelled cheaters) or cheater matches?"""
    neg = v[~v.is_cheater]
    out = []
    for name, acts in LEVELS.items():
        for src, g in neg.groupby("clean_match"):
            out.append(dict(level=name, negatives_from="clean matches" if src else "cheater matches",
                            negatives=len(g), flagged=int(g.action.isin(acts).sum()),
                            fpr=g.action.isin(acts).mean()))
    return pd.DataFrame(out)


def rule_table(v: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, text in RULE_TEXT.items():
        fired = v[v.fired_rules.map(lambda r: rid in r)]
        rows.append(dict(rule=rid, meaning=text, fired=len(fired),
                         share_cheaters=fired.is_cheater.mean() if len(fired) else np.nan))
    return pd.DataFrame(rows)


# ---------- figures -------------------------------------------------------------

def fig_pr_curve(v, ops):
    assessed = v[v.action != "no_action"]
    prec, rec = pr_curve(assessed.is_cheater, assessed.cheating_cf)
    prec, rec = np.r_[1.0, prec], np.r_[0.0, rec]
    base_rate = v.is_cheater.mean()
    # recall over ALL cheaters (unassessed cheaters can never be caught)
    rec_all = rec * assessed.is_cheater.sum() / v.is_cheater.sum()
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    # The final step is one block of tied players with CF = 0 (no rule fired): not a real trade-off.
    ax.plot(rec_all[:-1], prec[:-1], color=BLUE, lw=2, label="Expert system (cheating CF > 0)")
    ax.plot(rec_all[-2:], prec[-2:], color=BLUE, lw=1, ls="--", label="Also flagging CF = 0 (no rule fired)")
    ax.axhline(base_rate, color=INK2, lw=1, ls=":")
    ax.text(0.99, base_rate + 0.015, f"base rate {base_rate:.2f}", ha="right", color=INK2, fontsize=8)
    offsets = {  # hand-placed so labels never collide
        "ban review only": (0, 10, "center"), "hide or stricter": (0, -14, "center"),
        "monitor or stricter": (8, 6, "left"),
        "baseline: headshot rate > p95": (-8, -12, "right"),
        "baseline: any metric > p99": (8, -12, "left"), "baseline: any metric > p95": (8, -3, "left"),
    }
    for _, r in ops.iterrows():
        is_base = r.level.startswith("baseline")
        ax.scatter(r.recall, r.precision, s=46, zorder=3, color=ORANGE if is_base else BLUE,
                   marker="s" if is_base else "o", edgecolor="white", linewidth=1.5)
        dx, dy, ha = offsets[r.level]
        ax.annotate(r.level.replace("baseline: ", ""), (r.recall, r.precision), fontsize=7, ha=ha,
                    color=INK2, xytext=(dx, dy), textcoords="offset points")
    ax.scatter([], [], color=BLUE, marker="o", label="Expert-system actions")
    ax.scatter([], [], color=ORANGE, marker="s", label="Single-signal baselines")
    ax.set(xlabel="Recall (share of all labelled cheaters caught)", ylabel="Precision",
           xlim=(0, 1), ylim=(0, 1.02), title="Precision-recall of the expert system")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7.5)
    save(fig, "pr_curve")
    return float(np.trapezoid(prec, rec_all))


def fig_actions_by_label(v):
    fig, ax = plt.subplots(figsize=(6.2, 1.9))
    groups = [("Labelled cheaters", v[v.is_cheater]), ("Not labelled", v[~v.is_cheater])]
    for i, (name, g) in enumerate(groups):
        share = g.action.value_counts(normalize=True)
        left = 0.0
        for a in ACTION_ORDER:
            w = share.get(a, 0.0)
            ax.barh(i, w, left=left, color=ACTION_COLOR[a], edgecolor="white", linewidth=2, height=0.62,
                    label=ACTION_LABEL[a] if i == 0 else None)
            if w >= 0.06:
                ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if a in ("hide_from_leaderboard", "review_for_ban") else INK)
            left += w
    ax.set_yticks([0, 1], [f"{n}\n(n={len(g)})" for n, g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=7, handlelength=1)
    ax.set_title("Action taken by the expert system")
    save(fig, "actions_by_label")


def fig_cf_distribution(v):
    assessed = v[v.action != "no_action"]
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for name, g, c in [("Not labelled", assessed[~assessed.is_cheater], BLUE),
                       ("Labelled cheaters", assessed[assessed.is_cheater], ORANGE)]:
        ax.hist(g.cheating_cf, bins=bins, weights=np.ones(len(g)) / len(g), histtype="step",
                lw=2, color=c, label=f"{name} (n={len(g)})")
    for t, lab in [(0.2, "monitor"), (0.5, "hide"), (0.75, "ban review")]:
        ax.axvline(t, color=INK2, lw=0.8, ls=":")
        ax.text(t + 0.01, 0.9, lab, transform=ax.get_xaxis_transform(), fontsize=7, color=INK2)
    ax.set_yscale("log")
    ax.set(xlabel="Cheating certainty factor", ylabel="Share of players (log)",
           title="Certainty factor separates the two groups")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "cf_distribution")


def fig_calibration(values: pd.DataFrame, labels: pd.DataFrame, thresholds: pd.DataFrame):
    metrics = [("headshot_rate", "Headshot rate"), ("accuracy", "Accuracy"),
               ("wallbang_rate", "Wallbang kill rate"), ("adr", "Damage per round")]
    values = values.merge(labels, on="pid")
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4))
    for ax, (m, title) in zip(axes.flat, metrics):
        mv = values[values.metric == m]
        calib = mv[mv.match_id.map(is_calibration_match).astype(bool)].value
        cheat = mv[mv.is_cheater].value
        hi = max(calib.quantile(0.999), cheat.quantile(0.99))
        bins = np.linspace(0, hi, 30)
        for g, c, name in [(calib, BLUE, "Calibration players"), (cheat, ORANGE, "Labelled cheaters")]:
            ax.hist(g.clip(upper=hi), bins=bins, weights=np.ones(len(g)) / len(g), histtype="step",
                    lw=1.6, color=c, label=name)
        for lvl, ls in [("p95", "--"), ("p99", ":")]:
            ax.axvline(thresholds.loc[m, lvl], color=INK, lw=1, ls=ls)
        ax.set_title(title)
        ax.tick_params(labelsize=7)
    axes[0, 0].plot([], [], color=INK, ls="--", lw=1, label="p95 threshold")
    axes[0, 0].plot([], [], color=INK, ls=":", lw=1, label="p99 threshold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), fontsize=7.5)
    fig.supylabel("Share of players", fontsize=8, color=INK2)
    fig.suptitle("Thresholds set on normal play, compared with cheaters", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "calibration")


def fig_rules(rt: pd.DataFrame, base_rate: float):
    rt = rt[rt.fired > 0].sort_values("share_cheaters")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    y = np.arange(len(rt))
    ax.barh(y, rt.share_cheaters, color=BLUE, height=0.6)
    for i, r in enumerate(rt.itertuples()):
        ax.text(r.share_cheaters + 0.01, i, f"{r.share_cheaters:.0%}  (fired {r.fired}x)",
                va="center", fontsize=7, color=INK2)
    ax.axvline(base_rate, color=INK2, lw=1, ls="--")
    ax.text(base_rate + 0.01, len(rt) - 0.4, "base rate", fontsize=7, color=INK2)
    ax.set_yticks(y, [f"{r.rule}  {r.meaning}" for r in rt.itertuples()], fontsize=7.5)
    ax.set_xlim(0, 1.22)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(xlabel="Share of players the rule fired on who are labelled cheaters",
           title="How trustworthy is each rule?")
    ax.grid(axis="y", visible=False)
    save(fig, "rules")


def fig_dial(v):
    """Strictness dial: what happens as the ban-review threshold moves."""
    ts = np.round(np.arange(0.2, 1.0, 0.025), 3)
    y = v.is_cheater.to_numpy()
    rows = [dict(t=t, **scores(y, (v.cheating_cf >= t).to_numpy() & (v.action != "no_action").to_numpy())) for t in ts]
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(d.t, d.precision, color=BLUE, lw=2, label="Precision")
    ax.plot(d.t, d.recall, color=ORANGE, lw=2, label="Recall")
    ax.axvline(0.75, color=INK2, lw=0.8, ls=":")
    ax.text(0.755, 0.05, "default ban-review\nthreshold", fontsize=7, color=INK2)
    ax.set(xlabel="Ban-review threshold (cheating CF)", ylabel="Score", ylim=(0, 1.02),
           title="Strictness dial: ban-review threshold")
    ax.legend(fontsize=8)
    save(fig, "strictness_dial")
    return d


# ---------- tables ----------------------------------------------------------------

def write_tables(ops, fps, rt, diag):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    ops.to_csv(TAB_DIR / "operating_points.csv", index=False)
    fps.to_csv(TAB_DIR / "false_positives_by_source.csv", index=False)
    rt.to_csv(TAB_DIR / "rules.csv", index=False)
    diag.to_csv(TAB_DIR / "diagnosis_by_label.csv")

    t = ops[["level", "flagged", "precision", "recall", "f1", "fpr"]].copy()
    t.columns = ["Operating point", "Flagged", "Precision", "Recall", "F1", "FPR"]
    t.to_latex(TAB_DIR / "operating_points.tex", index=False, float_format="%.3f",
               caption="Expert system operating points against single-signal baselines (evaluation set).",
               label="tab:operating-points", escape=True)
    r = rt.copy()
    r["share_cheaters"] = r.share_cheaters.map(lambda x: "--" if pd.isna(x) else f"{x:.0%}")
    r.columns = ["Rule", "Meaning", "Fired", "Labelled cheaters"]
    r.to_latex(TAB_DIR / "rules.tex", index=False, escape=True,
               caption="Backward-chaining rules: how often each fired and the share of labelled cheaters among those players.",
               label="tab:rules")


def main():
    style()
    v = eval_set()
    ops = operating_points(v)
    fps = fp_by_source(v)
    rt = rule_table(v)
    diag = pd.crosstab(v.diagnosis, v.is_cheater.map({True: "labelled cheater", False: "not labelled"}))

    pr_auc = fig_pr_curve(v, ops)
    fig_actions_by_label(v)
    fig_cf_distribution(v)
    fig_rules(rt, v.is_cheater.mean())
    fig_dial(v)

    from cheatdetect import calibrate
    values = calibrate.metric_values()
    labels = pd.read_parquet(config.PLAYERS_FILE, columns=["pid", "is_cheater"])
    thresholds = pd.read_csv(config.REPORTS_DIR / "calibration_thresholds.csv", index_col=0)
    fig_calibration(values, labels, thresholds)

    write_tables(ops, fps, rt, diag)
    coverage = (v.action != "no_action").groupby(v.is_cheater).mean()
    metrics = dict(
        eval_players=len(v), eval_cheaters=int(v.is_cheater.sum()), base_rate=float(v.is_cheater.mean()),
        coverage_cheaters=float(coverage[True]), coverage_non_cheaters=float(coverage[False]),
        pr_auc=float(pr_auc),
        operating_points=ops.set_index("level")[["precision", "recall", "f1", "fpr", "flagged"]].to_dict("index"),
    )
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pd.set_option("display.width", 200)
    print(f"evaluation set: {len(v)} players, {v.is_cheater.sum()} labelled cheaters "
          f"(base rate {v.is_cheater.mean():.3f}); PR-AUC {pr_auc:.3f}")
    print(f"coverage (not 'insufficient evidence'): cheaters {coverage[True]:.1%}, others {coverage[False]:.1%}\n")
    print(ops[["level", "flagged", "precision", "recall", "f1", "fpr"]].round(3).to_string(index=False))
    print("\nfalse positives by source:\n", fps.round(3).to_string(index=False))
    print("\nrules:\n", rt.round(3).to_string(index=False))
    print("\ndiagnosis x label:\n", diag)


if __name__ == "__main__":
    main()
