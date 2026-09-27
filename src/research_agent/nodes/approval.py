"""人在回路（HITL）审批门。

本节点位于 planner 与并行扇出之间——在智能体为 N 条并发 researcher
分支花钱*之前*，由人工把关是再自然不过的位置。它使用 LangGraph 的
``interrupt()`` 原语：

* ``interrupt(payload)`` 会**暂停**整个图，并把 ``payload`` 呈现给调用方
  （操作员/UI）。当前状态已保存到检查点器，因此即使进程在这里死掉，
  之后也能恢复。
* 调用方检查载荷、取得人工决策，然后以 ``Command(resume=decision)`` 恢复。
  恢复时 LangGraph 会**从本节点顶部重新执行**，这一次 ``interrupt()``
  返回 ``decision`` 而不是暂停。

人工可以从四种操作中选择：

* ``approve`` —— 按原计划执行 → 流向扇出；
* ``edit``    —— 用人工修改后的子任务列表替换 → 扇出；
* ``reject``  —— 退回 planner 重新起草，可附带反馈；
* ``abort``   —— 彻底停止运行 → 直接路由到 END，并写入中止报告。

由于 ``interrupt()`` 需要持久化状态，只有在配置了检查点器且
``settings.hitl_approval_enabled`` 为 True 时，本门才会真正起作用。
关闭时它是透明的直通节点，因此评估/CI/API 默认路径保持完全自主。
"""

from __future__ import annotations

from langgraph.types import interrupt

from research_agent.config import settings


def approval_node(state):
    """暂停等待人工审批第一份计划；其余情况直接放行。"""
    # 完全跳过审批门（返回干净的、未被拒绝的状态）有两个原因：
    #   1. HITL 已关闭 → 保持自主。
    #   2. 正处于 critic 回环的重新规划（iterations > 0）→ 人工已经批准过
    #      本次运行的方向；不要每个循环都去烦他们。只有第一次开销大的
    #      扇出才需要签字。
    if not settings.hitl_approval_enabled or state.get("iterations", 0) > 0:
        return {"_plan_rejected": False}

    # 把人工正在审批的内容精确地快照进 interrupt 载荷，让操作员看清
    # 自己究竟在批准什么（最佳实践：绝不让人批准自己看不见的东西）。
    decision = interrupt(
        {
            "type": "plan_approval",
            "message": (
                "请审阅拟定研究计划后继续。可选操作："
                "{'action': 'approve'}（通过）| "
                "{'action': 'edit', 'subtasks': [...]}（修改子任务）| "
                "{'action': 'reject', 'feedback': '...'}（打回重规划）| "
                "{'action': 'abort', 'reason': '...'}（中止）"
            ),
            "query": state.get("query"),
            "proposed_subtasks": state.get("subtasks", []),
        }
    )

    # ── 以下代码只在人工恢复之后才会执行 ──────────────────────────────────────
    # `decision` 是传给 Command(resume=...) 的任何内容。默认取
    # "approve"，使空的恢复调用被视为放行。
    action = (decision or {}).get("action", "approve")

    if action == "edit":
        # 人工给出了修正后的子任务列表 → 采用它，然后扇出。
        return {"subtasks": decision["subtasks"], "_plan_rejected": False, "plan_feedback": ""}

    if action == "reject":
        # 路由回 planner 重新起草。其自由文本反馈（如有）会被保存，
        # 供 planner 在下一轮处理。
        return {"_plan_rejected": True, "plan_feedback": decision.get("feedback", "")}

    if action == "abort":
        # 彻底停止运行。直接路由到 END（见 route_after_approval），
        # 并写入报告形状的结果，使调用方获得统一的输出，与输入护栏
        # 拒绝查询的方式一致。图会到达终态，因此最终检查点记录的是
        # 中止，而不是留下一个悬而未决的暂停。
        reason = decision.get("reason") or "操作员在审批环节中止了本次运行。"
        return {
            "_aborted": True,
            "report": {"summary": f"运行已中止：{reason}", "sections": [], "citations": []},
        }

    # approve（或未知操作）→ 按原样继续。仍然显式把标志设为 False，
    # 防止先前 reject 遗留的过期 `True` 残留在持久化状态中，
    # 再次触发拒绝路由。
    return {"_plan_rejected": False, "plan_feedback": ""}
