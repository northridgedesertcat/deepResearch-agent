"""评估测试装置共享的 pytest 配置。

放在仓库根目录，使根目录位于 ``sys.path`` 上（让 ``evals`` 可以与已安装的
``research_agent`` 包一起导入），并确保"在整个数据集上运行智能体"这一开销
大的步骤在每个会话中只执行一次。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

import pytest
from dotenv import load_dotenv

from research_agent.state import ResearchState

load_dotenv()

_DATASET = Path(__file__).parent / "evals" / "dataset.json"


def pytest_configure(config):
    config.addinivalue_line("markers", "eval: 端到端智能体质量评估（慢，消耗 LLM 调用）")


def _load_cases():
    cases = json.loads(_DATASET.read_text(encoding="utf-8"))
    from research_agent.config import settings

    if settings.eval_max_cases > 0:
        cases = cases[: settings.eval_max_cases]
    return cases


@pytest.fixture(scope="session")
def eval_results():
    """对每个数据集用例各运行一次智能体，并对每个输出打分。

    返回一个字典列表：id、faithfulness、citation_accuracy、coverage，
    以及运行指标（latency_s、iterations）。未配置 API key 时跳过整个评估，
    使测试套件在非 CI 环境下优雅降级。
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping live agent evals")

    from research_agent.graph import build_graph
    from evals.judges import faithfulness, citation_accuracy, coverage

    app = build_graph()
    results = []
    for case in _load_cases():
        start = time.perf_counter()

        state = app.invoke(cast(ResearchState, {"query": case["query"]}))
        latency_s = time.perf_counter() - start

        findings = state.get("findings", [])
        report = state.get("report", {})

        results.append(
            {
                "id": case["id"],
                "faithfulness": faithfulness(findings, report).score,
                "citation_accuracy": citation_accuracy(findings, report),
                "coverage": coverage(case["query"], case["key_points"], report).score,
                "latency_s": round(latency_s, 1),
                "iterations": state.get("iterations", 0),
            }
        )
    return results
