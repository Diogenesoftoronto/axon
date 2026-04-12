#!/usr/bin/env python3
"""Bayesian inference, Thompson sampling, and latency variability analysis for Axon benchmarks.

Reads results JSON files and produces:
  - Beta-Binomial posterior analysis for pass rates
  - Thompson sampling (multi-armed bandit) mode selection
  - Latency variability statistics (mean, std, CV, quartiles)
  - LaTeX-ready output for paper figures/tables

Usage:
  python3 scripts/analyze_bayesian.py benchmarks/results/results-pareto-multimodel.json [more.json ...]
  python3 scripts/analyze_bayesian.py --from-log /tmp/full-5profile.log
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Beta distribution helpers (no scipy needed)
# ---------------------------------------------------------------------------

def beta_mean(a: float, b: float) -> float:
    return a / (a + b)

def beta_variance(a: float, b: float) -> float:
    return (a * b) / ((a + b) ** 2 * (a + b + 1))

def beta_pdf(x: float, a: float, b: float) -> float:
    if x <= 0 or x >= 1:
        return 0.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp((a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - log_beta)

def beta_cdf_approx(x: float, a: float, b: float, steps: int = 1000) -> float:
    """Numerical integration of Beta CDF via trapezoidal rule."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    dx = x / steps
    total = 0.0
    for i in range(steps + 1):
        xi = i * dx
        w = 1.0 if (0 < i < steps) else 0.5
        total += w * beta_pdf(xi, a, b)
    return total * dx

def beta_quantile(p: float, a: float, b: float, tol: float = 1e-6) -> float:
    """Binary search for quantile of Beta distribution."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if beta_cdf_approx(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2

def beta_hdi(a: float, b: float, credible: float = 0.95) -> Tuple[float, float]:
    """Approximate highest-density interval via equal-tailed credible interval."""
    alpha = (1.0 - credible) / 2.0
    lo = beta_quantile(alpha, a, b)
    hi = beta_quantile(1.0 - alpha, a, b)
    return (lo, hi)

def prob_a_greater_b(a1: float, b1: float, a2: float, b2: float,
                     samples: int = 50000) -> float:
    """Monte Carlo estimate of P(X > Y) where X ~ Beta(a1,b1), Y ~ Beta(a2,b2)."""
    import random
    random.seed(42)
    count = 0
    for _ in range(samples):
        x = random.betavariate(a1, b1)
        y = random.betavariate(a2, b2)
        if x > y:
            count += 1
    return count / samples


# ---------------------------------------------------------------------------
# Thompson Sampling simulation
# ---------------------------------------------------------------------------

def thompson_sampling(arms: Dict[str, Tuple[float, float]],
                      n_rounds: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Simulate Thompson sampling selection frequency.

    arms: {arm_name: (alpha, beta)} Beta posterior parameters
    Returns: {arm_name: selection_fraction}
    """
    import random
    random.seed(seed)
    counts = {k: 0 for k in arms}
    for _ in range(n_rounds):
        best_arm = None
        best_val = -1.0
        for name, (a, b) in arms.items():
            sample = random.betavariate(a, b)
            if sample > best_val:
                best_val = sample
                best_arm = name
        counts[best_arm] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    model: str
    mode: str
    task_id: str
    ok: bool
    matched: bool
    elapsed_s: float
    tokens: int
    cost: float

def load_results_json(path: Path) -> List[RunRecord]:
    data = json.loads(path.read_text())
    records = []
    for r in data.get("results", []):
        records.append(RunRecord(
            model=r.get("model", ""),
            mode=r.get("mode", ""),
            task_id=r.get("task_id", ""),
            ok=r.get("ok", False),
            matched=r.get("matched", False),
            elapsed_s=r.get("elapsed_s", 0.0),
            tokens=r.get("total_tokens", 0),
            cost=r.get("estimated_cost_usd", 0.0),
        ))
    return records

def load_from_log(path: Path) -> List[RunRecord]:
    """Parse benchmark log output for interim results."""
    pat = re.compile(
        r"\[(?P<model>[^\]]+)\]\[(?P<mode>[^\]]+)\] (?P<task>\S+) run=\d+ -> "
        r"(?P<status>PASS|FAIL|ERROR) \((?P<time>[\d.]+)s\)"
    )
    records = []
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            status = m.group("status")
            records.append(RunRecord(
                model=m.group("model"),
                mode=m.group("mode"),
                task_id=m.group("task"),
                ok=(status != "ERROR"),
                matched=(status == "PASS"),
                elapsed_s=float(m.group("time")),
                tokens=0,
                cost=0.0,
            ))
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def group_by_model_mode(records: List[RunRecord]) -> Dict[Tuple[str, str], List[RunRecord]]:
    groups: Dict[Tuple[str, str], List[RunRecord]] = defaultdict(list)
    for r in records:
        groups[(r.model, r.mode)].append(r)
    return dict(groups)


def short_model(name: str) -> str:
    s = name.split("/")[-1]
    replacements = {
        "MiniMax-M2.5": "M2.5",
        "MiniMax-M2.1": "M2.1",
        "Qwen3.5-397B-A17B": "Qwen3.5",
        "Qwen3-235B-A22B-Thinking-2507": "Qwen3-235B",
        "DeepSeek-V3.2": "DSV3.2",
        "Kimi-K2-Thinking": "Kimi-K2",
        "Kimi-K2.5": "K2.5",
    }
    return replacements.get(s, s)


def short_mode(name: str) -> str:
    replacements = {
        "d0-i1": "d0-i1",
        "d0-i3": "d0-i3",
        "d1-i3": "d1-i3",
        "d3-i1": "d3-i1",
        "d6-i1": "d6-i1",
        "current-default": "d0-i1",
        "current-no-recursion-best-of-3": "d0-i3",
        "current-depth1-iter3": "d1-i3",
        "current-depth6-single-pass": "d6-i1",
    }
    return replacements.get(name, name)


def bayesian_analysis(groups: Dict[Tuple[str, str], List[RunRecord]],
                      exclude_error_models: bool = True) -> None:
    """Compute and print Beta-Binomial posterior analysis."""
    print("\n" + "=" * 80)
    print("BAYESIAN POSTERIOR ANALYSIS (Beta-Binomial)")
    print("Prior: Beta(1,1) = Uniform")
    print("=" * 80)

    posteriors: Dict[Tuple[str, str], Tuple[float, float, int, int]] = {}

    for (model, mode), runs in sorted(groups.items()):
        n = len(runs)
        ok_runs = [r for r in runs if r.ok]
        n_ok = len(ok_runs)

        if exclude_error_models and n_ok == 0 and n > 0:
            # All errors — likely model down, skip
            continue

        k = sum(1 for r in runs if r.matched)
        # Beta(1,1) prior -> Beta(1+k, 1+n-k) posterior
        a_post = 1 + k
        b_post = 1 + n - k
        mean = beta_mean(a_post, b_post)
        lo, hi = beta_hdi(a_post, b_post, 0.95)

        posteriors[(model, mode)] = (a_post, b_post, k, n)

        sm = short_model(model)
        smode = short_mode(mode)
        print(f"  {sm:15s} {smode:8s}  {k}/{n} pass  "
              f"Post Beta({a_post},{b_post})  "
              f"mean={mean:.3f}  95% HDI=[{lo:.3f}, {hi:.3f}]")

    # Pairwise superiority for same model
    print("\n--- Pairwise P(mode_A > mode_B) per model ---")
    models = sorted(set(m for m, _ in posteriors))
    modes = sorted(set(md for _, md in posteriors))

    for model in models:
        sm = short_model(model)
        model_arms = {short_mode(md): posteriors[(model, md)]
                      for md in modes if (model, md) in posteriors}
        if len(model_arms) < 2:
            continue
        print(f"\n  {sm}:")
        arm_names = sorted(model_arms.keys())
        for i, a1 in enumerate(arm_names):
            for a2 in arm_names[i + 1:]:
                p1 = model_arms[a1]
                p2 = model_arms[a2]
                prob = prob_a_greater_b(p1[0], p1[1], p2[0], p2[1])
                print(f"    P({a1} > {a2}) = {prob:.3f}")

    return posteriors


def thompson_analysis(groups: Dict[Tuple[str, str], List[RunRecord]],
                      exclude_error_models: bool = True) -> None:
    """Multi-armed bandit Thompson sampling analysis per model."""
    print("\n" + "=" * 80)
    print("THOMPSON SAMPLING MODE SELECTION")
    print("10,000 rounds of Thompson sampling per model")
    print("=" * 80)

    models = sorted(set(m for m, _ in groups))
    modes = sorted(set(md for _, md in groups))

    all_model_results = {}

    for model in models:
        sm = short_model(model)
        arms = {}
        for mode in modes:
            key = (model, mode)
            if key not in groups:
                continue
            runs = groups[key]
            n = len(runs)
            if exclude_error_models and all(not r.ok for r in runs):
                continue
            k = sum(1 for r in runs if r.matched)
            a_post = 1 + k
            b_post = 1 + n - k
            arms[short_mode(mode)] = (a_post, b_post)

        if len(arms) < 2:
            continue

        freq = thompson_sampling(arms)
        all_model_results[sm] = freq
        print(f"\n  {sm}:")
        for arm, f in sorted(freq.items(), key=lambda x: -x[1]):
            bar = "█" * int(f * 40)
            print(f"    {arm:8s}: {f:.3f}  {bar}")

    # Cross-model aggregation: which mode wins most often?
    print("\n--- Aggregate mode selection frequency (macro-avg across models) ---")
    mode_totals: Dict[str, float] = defaultdict(float)
    n_models = len(all_model_results)
    for model_freq in all_model_results.values():
        for mode, f in model_freq.items():
            mode_totals[mode] += f
    for mode in sorted(mode_totals, key=lambda m: -mode_totals[m]):
        avg = mode_totals[mode] / n_models
        bar = "█" * int(avg * 40)
        print(f"    {mode:8s}: {avg:.3f}  {bar}")

    return all_model_results


def latency_analysis(groups: Dict[Tuple[str, str], List[RunRecord]],
                     exclude_error_models: bool = True) -> None:
    """Latency variability analysis."""
    print("\n" + "=" * 80)
    print("LATENCY VARIABILITY ANALYSIS")
    print("=" * 80)
    print(f"  {'Model':15s} {'Mode':8s} {'N':>3s} {'Mean':>7s} {'Std':>7s} {'CV':>6s} "
          f"{'Min':>7s} {'Q1':>7s} {'Med':>7s} {'Q3':>7s} {'Max':>7s}")
    print("  " + "-" * 85)

    for (model, mode), runs in sorted(groups.items()):
        if exclude_error_models and all(not r.ok for r in runs):
            continue
        times = sorted(r.elapsed_s for r in runs if r.ok)
        if not times:
            continue
        n = len(times)
        mean = sum(times) / n
        var = sum((t - mean) ** 2 for t in times) / n if n > 1 else 0
        std = var ** 0.5
        cv = std / mean if mean > 0 else 0
        q1 = times[n // 4] if n >= 4 else times[0]
        med = times[n // 2]
        q3 = times[3 * n // 4] if n >= 4 else times[-1]

        sm = short_model(model)
        smode = short_mode(mode)
        print(f"  {sm:15s} {smode:8s} {n:3d} {mean:7.1f} {std:7.1f} {cv:6.2f} "
              f"{times[0]:7.1f} {q1:7.1f} {med:7.1f} {q3:7.1f} {times[-1]:7.1f}")


def latex_bayesian_table(groups: Dict[Tuple[str, str], List[RunRecord]],
                         exclude_error_models: bool = True) -> str:
    """Generate LaTeX table for Bayesian posteriors."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Bayesian posterior analysis: Beta-Binomial model with uniform prior. $k$=passes, $n$=trials, posterior Beta$(1{+}k, 1{+}n{-}k)$.}",
        r"\label{tab:bayesian-posteriors}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Mode & $k/n$ & Post.\ mean & 95\% HDI low & 95\% HDI high \\",
        r"\midrule",
    ]

    for (model, mode), runs in sorted(groups.items()):
        if exclude_error_models and all(not r.ok for r in runs):
            continue
        n = len(runs)
        k = sum(1 for r in runs if r.matched)
        a, b = 1 + k, 1 + n - k
        mean = beta_mean(a, b)
        lo, hi = beta_hdi(a, b, 0.95)
        sm = short_model(model)
        smode = short_mode(mode)
        lines.append(f"  {sm} & {smode} & {k}/{n} & {mean:.3f} & {lo:.3f} & {hi:.3f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_thompson_table(all_model_results: Dict[str, Dict[str, float]]) -> str:
    """Generate LaTeX table for Thompson sampling results."""
    modes = sorted(set(m for freq in all_model_results.values() for m in freq))
    mode_header = " & ".join(modes)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Thompson sampling selection probability per model (10{,}000 rounds). Higher values indicate the mode is more likely to be optimal under posterior uncertainty.}",
        r"\label{tab:thompson-sampling}",
        r"\begin{tabular}{l" + "r" * len(modes) + "}",
        r"\toprule",
        f"Model & {mode_header} \\\\",
        r"\midrule",
    ]

    for model in sorted(all_model_results):
        freq = all_model_results[model]
        vals = []
        best_val = max(freq.values())
        for m in modes:
            v = freq.get(m, 0.0)
            s = f"{v:.3f}"
            if v == best_val:
                s = r"\textbf{" + s + "}"
            vals.append(s)
        lines.append(f"  {model} & {' & '.join(vals)} \\\\")

    # Aggregate row
    n_models = len(all_model_results)
    agg = defaultdict(float)
    for freq in all_model_results.values():
        for m, v in freq.items():
            agg[m] += v
    agg_vals = []
    for m in modes:
        agg_vals.append(f"{agg.get(m, 0) / n_models:.3f}")
    lines.append(r"\midrule")
    lines.append(f"  Macro-avg & {' & '.join(agg_vals)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_latency_table(groups: Dict[Tuple[str, str], List[RunRecord]],
                        exclude_error_models: bool = True) -> str:
    """Generate LaTeX table for latency variability."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Latency variability per model-mode (seconds). CV = coefficient of variation (std/mean).}",
        r"\label{tab:latency-variability}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Mode & $n$ & Mean & Std & CV & Min & Max \\",
        r"\midrule",
    ]

    for (model, mode), runs in sorted(groups.items()):
        if exclude_error_models and all(not r.ok for r in runs):
            continue
        times = [r.elapsed_s for r in runs if r.ok]
        if not times:
            continue
        n = len(times)
        mean = sum(times) / n
        var = sum((t - mean) ** 2 for t in times) / n if n > 1 else 0
        std = var ** 0.5
        cv = std / mean if mean > 0 else 0

        sm = short_model(model)
        smode = short_mode(mode)
        lines.append(f"  {sm} & {smode} & {n} & {mean:.1f} & {std:.1f} & {cv:.2f} & {min(times):.1f} & {max(times):.1f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_bayesian_ci_figure(groups: Dict[Tuple[str, str], List[RunRecord]],
                             exclude_error_models: bool = True) -> str:
    """Generate pgfplots figure showing Bayesian 95% credible intervals per model-mode."""
    # Collect data points: (model, mode, mean, lo, hi)
    points = []
    for (model, mode), runs in sorted(groups.items()):
        if exclude_error_models and all(not r.ok for r in runs):
            continue
        n = len(runs)
        k = sum(1 for r in runs if r.matched)
        a, b = 1 + k, 1 + n - k
        mean = beta_mean(a, b) * 100
        lo, hi = beta_hdi(a, b, 0.95)
        lo *= 100
        hi *= 100
        points.append((short_model(model), short_mode(mode), mean, lo, hi))

    # Group by model for plotting
    models = list(dict.fromkeys(p[0] for p in points))
    modes = list(dict.fromkeys(p[1] for p in points))

    # Generate error bar plot: one subplot per mode, models on x-axis
    colors = ["blue", "red", "green!60!black", "orange", "purple"]

    lines = [
        r"\begin{figure}[t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.95\linewidth,",
        r"  height=8cm,",
        r"  ymin=-5, ymax=105,",
        r"  ylabel={Posterior mean pass rate (\%) with 95\% HDI},",
        r"  symbolic x coords={" + ",".join(models) + "},",
        r"  xtick=data,",
        r"  x tick label style={rotate=25,anchor=east,font=\small},",
        r"  legend style={at={(0.5,1.15)},anchor=south,legend columns=" + str(min(len(modes), 5)) + r",font=\small},",
        r"  grid=major,",
        r"]",
    ]

    for i, mode in enumerate(modes):
        color = colors[i % len(colors)]
        coords = []
        for model in models:
            match = [p for p in points if p[0] == model and p[1] == mode]
            if match:
                _, _, mean, lo, hi = match[0]
                err_lo = mean - lo
                err_hi = hi - mean
                coords.append(f"({model},{mean}) +- (0,{err_hi}) +- (0,{err_lo})")

        if coords:
            shift = (i - len(modes) / 2 + 0.5) * 3
            lines.append(f"\\addplot[{color},mark=*,mark size=2pt,error bars/.cd,y dir=both,y explicit,")
            lines.append(f"  error bar style={{{color}!60}},error mark options={{rotate=90,mark size=3pt,{color}}}]")
            # Use explicit error bars
            coord_strs = []
            for model in models:
                match = [p for p in points if p[0] == model and p[1] == mode]
                if match:
                    _, _, mean, lo, hi = match[0]
                    coord_strs.append(f"  ({model},{mean:.1f}) +- (0,{hi - mean:.1f})")
            lines.append("  coordinates {")
            lines.append("    " + "\n    ".join(coord_strs))
            lines.append("  };")

    lines.append(r"\legend{" + ",".join(modes) + "}")
    lines += [r"\end{axis}", r"\end{tikzpicture}",
              r"\caption{Bayesian 95\% highest-density intervals for pass rate by model and mode. "
              r"Points show posterior mean; bars show 95\% credible interval under Beta(1,1) prior. "
              r"Wide intervals reflect $n{=}8$ trials per configuration.}",
              r"\label{fig:bayesian-hdi}",
              r"\end{figure}"]
    return "\n".join(lines)


def latex_thompson_figure(all_model_results: Dict[str, Dict[str, float]]) -> str:
    """Generate pgfplots stacked bar chart for Thompson sampling allocation."""
    models = sorted(all_model_results.keys())
    modes = sorted(set(m for freq in all_model_results.values() for m in freq))
    colors = ["blue!70", "red!70", "green!50!black", "orange!80", "purple!70"]

    lines = [
        r"\begin{figure}[t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  ybar stacked,",
        r"  bar width=12pt,",
        r"  width=0.95\linewidth,",
        r"  height=6.5cm,",
        r"  ymin=0, ymax=1.05,",
        r"  ylabel={Thompson sampling selection probability},",
        r"  symbolic x coords={" + ",".join(models) + "},",
        r"  xtick=data,",
        r"  x tick label style={rotate=20,anchor=east,font=\small},",
        r"  legend style={at={(0.5,1.15)},anchor=south,legend columns=" + str(min(len(modes), 5)) + r",font=\small},",
        r"]",
    ]

    for i, mode in enumerate(modes):
        color = colors[i % len(colors)]
        coords = []
        for model in models:
            v = all_model_results[model].get(mode, 0.0)
            coords.append(f"({model},{v:.4f})")
        lines.append(f"\\addplot[fill={color}] coordinates {{{' '.join(coords)}}};")

    lines.append(r"\legend{" + ",".join(modes) + "}")
    lines += [r"\end{axis}", r"\end{tikzpicture}",
              r"\caption{Thompson sampling mode allocation per model (10{,}000 rounds). "
              r"Each bar shows the probability that Thompson sampling would select each mode as optimal. "
              r"Concentrated bars indicate high confidence; diffuse bars indicate mode-insensitivity.}",
              r"\label{fig:thompson-allocation}",
              r"\end{figure}"]
    return "\n".join(lines)


def latex_latency_cv_figure(groups: Dict[Tuple[str, str], List[RunRecord]],
                            exclude_error_models: bool = True) -> str:
    """Generate scatter plot of mean latency vs CV per model-mode."""
    points = []
    for (model, mode), runs in sorted(groups.items()):
        if exclude_error_models and all(not r.ok for r in runs):
            continue
        times = [r.elapsed_s for r in runs if r.ok]
        if not times or len(times) < 2:
            continue
        n = len(times)
        mean = sum(times) / n
        std = (sum((t - mean) ** 2 for t in times) / n) ** 0.5
        cv = std / mean if mean > 0 else 0
        points.append((short_model(model), short_mode(mode), mean, cv))

    if not points:
        return "% No latency data available for CV figure"

    modes = list(dict.fromkeys(p[1] for p in points))
    colors = ["blue", "red", "green!60!black", "orange", "purple"]
    marks = ["*", "square*", "triangle*", "diamond*", "pentagon*"]

    lines = [
        r"\begin{figure}[t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.95\linewidth,",
        r"  height=6.5cm,",
        r"  xlabel={Mean latency (s)},",
        r"  ylabel={Coefficient of variation (std/mean)},",
        r"  grid=major,",
        r"  legend style={at={(0.03,0.97)},anchor=north west,font=\small},",
        r"]",
    ]

    for i, mode in enumerate(modes):
        color = colors[i % len(colors)]
        mark = marks[i % len(marks)]
        mode_pts = [p for p in points if p[1] == mode]
        coords = " ".join(f"({p[2]:.1f},{p[3]:.3f})" for p in mode_pts)
        lines.append(f"\\addplot[only marks,mark={mark},mark size=3pt,{color}] coordinates {{{coords}}};")
        # Labels
        for p in mode_pts:
            lines.append(f"\\node[anchor=south west,font=\\scriptsize,{color}] at (axis cs:{p[2]:.1f},{p[3]:.3f}) {{{p[0]}}};")

    lines.append(r"\legend{" + ",".join(modes) + "}")
    lines += [r"\end{axis}", r"\end{tikzpicture}",
              r"\caption{Latency mean vs coefficient of variation by model and mode. "
              r"High CV indicates unstable response times. Ideal configurations are in the lower-left quadrant.}",
              r"\label{fig:latency-cv}",
              r"\end{figure}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bayesian and MAB analysis for Axon benchmarks")
    parser.add_argument("files", nargs="*", help="Results JSON files")
    parser.add_argument("--from-log", type=Path, default=None, help="Parse benchmark log instead")
    parser.add_argument("--exclude-error-models", action="store_true", default=True,
                        help="Exclude model-mode combos with 100%% errors")
    parser.add_argument("--latex-out", type=Path, default=None,
                        help="Write LaTeX snippets to this file")
    args = parser.parse_args()

    records: List[RunRecord] = []
    if args.from_log:
        records = load_from_log(args.from_log)
        print(f"Loaded {len(records)} records from log: {args.from_log}")
    for f in args.files:
        r = load_results_json(Path(f))
        records.extend(r)
        print(f"Loaded {len(r)} records from {f}")

    if not records:
        print("No records loaded", file=sys.stderr)
        return 1

    groups = group_by_model_mode(records)
    print(f"\nTotal: {len(records)} runs across {len(groups)} model-mode configurations")

    # Run analyses
    posteriors = bayesian_analysis(groups, args.exclude_error_models)
    thompson_results = thompson_analysis(groups, args.exclude_error_models)
    latency_analysis(groups, args.exclude_error_models)

    # Generate LaTeX
    latex_parts = []
    latex_parts.append("% === Bayesian Posterior Table ===")
    latex_parts.append(latex_bayesian_table(groups, args.exclude_error_models))
    latex_parts.append("")
    latex_parts.append("% === Bayesian HDI Figure ===")
    latex_parts.append(latex_bayesian_ci_figure(groups, args.exclude_error_models))
    latex_parts.append("")
    if thompson_results:
        latex_parts.append("% === Thompson Sampling Table ===")
        latex_parts.append(latex_thompson_table(thompson_results))
        latex_parts.append("")
        latex_parts.append("% === Thompson Sampling Figure ===")
        latex_parts.append(latex_thompson_figure(thompson_results))
        latex_parts.append("")
    latex_parts.append("% === Latency Variability Table ===")
    latex_parts.append(latex_latency_table(groups, args.exclude_error_models))
    latex_parts.append("")
    latex_parts.append("% === Latency CV Figure ===")
    latex_parts.append(latex_latency_cv_figure(groups, args.exclude_error_models))

    latex_output = "\n".join(latex_parts)

    if args.latex_out:
        args.latex_out.write_text(latex_output)
        print(f"\nLaTeX snippets written to {args.latex_out}")
    else:
        print("\n" + "=" * 80)
        print("LATEX OUTPUT")
        print("=" * 80)
        print(latex_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
