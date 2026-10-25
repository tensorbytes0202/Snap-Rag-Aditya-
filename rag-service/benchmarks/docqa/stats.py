"""Statistics for the benchmark reports: exact and bootstrap intervals, paired tests, multiplicity.

    exact_interval      Clopper-Pearson 95% interval of a proportion (k successes in n)
    bootstrap_ci        percentile bootstrap interval of a mean (continuous scores)
    paired_p            p-value of a paired difference: exact sign test for 0/1 outcomes,
                        Wilcoxon signed-rank test for scores
    holm                Holm-Bonferroni adjusted p-values for a family of comparisons
    auroc               how well a confidence separates right answers from wrong ones
"""

import math
import random

from benchmarks.longconv.report import bootstrap_ci

__all__ = ["bootstrap_ci", "exact_interval", "is_binary", "sign_p", "paired_p", "holm", "auroc", "bootstrap_diff"]


def exact_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson interval; exact, so it never collapses to a single point
    for a proportion of 0 or 1 the way a bootstrap does."""

    from scipy.stats import beta

    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))

    return low, high


def is_binary(values) -> bool:

    return bool(values) and all(v in (0, 1) for v in values)


def sign_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value (ties are left out)."""

    n = wins + losses

    if n == 0:
        return 1.0

    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n

    return min(1.0, 2 * tail)


def paired_p(diffs: list[float]) -> float:
    """p-value that a paired difference is zero: the exact sign test when every
    difference is -1, 0 or +1 (0/1 outcomes), Wilcoxon signed-rank otherwise."""

    if all(d == 0 for d in diffs):
        return 1.0

    if all(d in (-1, 0, 1) for d in diffs):
        return sign_p(sum(d > 0 for d in diffs), sum(d < 0 for d in diffs))

    from scipy.stats import wilcoxon

    return float(wilcoxon(diffs, zero_method="wilcox").pvalue)


def holm(p_values: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for {name: p}; controls the chance of any
    false positive across the whole family."""

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total, running, adjusted = len(ordered), 0.0, {}

    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p))
        adjusted[name] = running

    return adjusted


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Probability that a right answer has a higher score than a wrong one (0.5 = no signal);
    None when only one kind of answer exists."""

    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]

    if not pos or not neg:
        return None

    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)

    return wins / (len(pos) * len(neg))


def bootstrap_diff(a: list[float], b: list[float], iterations: int = 2000, seed: int = 0):
    """Percentile bootstrap interval of mean(a) - mean(b) over resampled questions (paired)."""

    rng = random.Random(seed)
    n = len(a)
    diffs = []

    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)

    diffs.sort()

    return diffs[int(0.025 * iterations)], diffs[int(0.975 * iterations) - 1]
