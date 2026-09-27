"""
CI 门控的智能体质量评估。

在整个评估数据集上运行智能体，用 ``evals/judges.py`` 中的评审给每个输出
打分，若平均 faithfulness / citation accuracy / coverage 低于 ``config.py``
中的阈值则失败。这是在每次 pull request 上运行的回归门
（见 ``.github/workflows/evals.yml``）。

本地运行：     uv run pytest -m eval -s
低成本子集：   EVAL_MAX_CASES=3 uv run pytest -m eval -s
"""

from __future__ import annotations

from statistics import mean

import pytest

from research_agent.config import settings

pytestmark = pytest.mark.eval


def _mean(results, key: str) -> float:
    return mean(r[key] for r in results)


def test_print_report(eval_results):
    """不是门槛——打印每个用例的记分卡，让 CI 日志能看到发生了什么。"""
    header = f"{'case':24} {'faith':>6} {'cite':>6} {'cover':>6} {'lat(s)':>7} {'iters':>5}"
    print("\n" + header)
    print("-" * len(header))
    for r in eval_results:
        print(
            f"{r['id']:24} {r['faithfulness']:6.2f} {r['citation_accuracy']:6.2f} "
            f"{r['coverage']:6.2f} {r['latency_s']:7.1f} {r['iterations']:5d}"
        )
    print("-" * len(header))
    print(
        f"{'MEAN':24} {_mean(eval_results, 'faithfulness'):6.2f} "
        f"{_mean(eval_results, 'citation_accuracy'):6.2f} "
        f"{_mean(eval_results, 'coverage'):6.2f}"
    )


def test_faithfulness_gate(eval_results):
    score = _mean(eval_results, "faithfulness")
    assert score >= settings.eval_min_faithfulness, (
        f"mean faithfulness {score:.2f} < threshold {settings.eval_min_faithfulness}"
    )


def test_citation_accuracy_gate(eval_results):
    score = _mean(eval_results, "citation_accuracy")
    assert score >= settings.eval_min_citation_accuracy, (
        f"mean citation accuracy {score:.2f} < threshold {settings.eval_min_citation_accuracy}"
    )


def test_coverage_gate(eval_results):
    score = _mean(eval_results, "coverage")
    assert score >= settings.eval_min_coverage, (
        f"mean coverage {score:.2f} < threshold {settings.eval_min_coverage}"
    )
