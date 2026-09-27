"""研究图的持久化检查点。

**检查点器（checkpointer）**在每个超级步之后持久化完整的图状态，以
``thread_id`` 为键。仅这一项改动就让图变得可持久化，并同时解锁四个
一等特性：

* **暂停/恢复** —— 运行中途停下，之后用同一个 ``thread_id`` 继续；
* **崩溃恢复** —— 进程死亡后，最后一个检查点仍在磁盘上；
* **时间旅行** —— 从任意历史检查点重放图，便于调试；
* **人在回路** —— ``interrupt()`` 之所以能暂停/恢复，正是因为状态
  在暂停与人工回复之间被持久化保存了。

本模块是决定*由哪个*后端存储这些检查点的唯一位置。代码库其余部分只管
调用 ``build_graph(checkpointer=...)``，从不关心存储细节——这正是
检查点器抽象的全部意义。

--------------------------------------------------------------------------------
开发环境（本项目所用）：SqliteSaver
--------------------------------------------------------------------------------
单个本地文件（默认 ``checkpoints.db``）。零配置、零成本，非常适合开发、
单进程服务器以及本项目的范围。

    from research_agent.persistence import sqlite_saver
    from research_agent.graph import build_graph

    with sqlite_saver() as saver:                 # 打开 SQLite 连接
        app = build_graph(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "run-123"}}
        app.invoke({"query": "..."}, config=cfg)  # 状态自此被持久化

--------------------------------------------------------------------------------
生产环境：请改用 PostgresSaver
--------------------------------------------------------------------------------
SQLite 是单文件、单写入者的存储。真实部署中多个 worker/进程同时写同一
数据库时需要真正的数据库。LangGraph 官方支持的生产后端是 Postgres，而且
——关键在于——*唯一*需要改变的就是 saver；图代码完全不变。

    # pip install langgraph-checkpoint-postgres
    import os
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as saver:
        saver.setup()                             # 首次运行：建表
        app = build_graph(checkpointer=saver)
        app.invoke({"query": "..."}, config={"configurable": {"thread_id": "run-123"}})

任何免费的 Postgres 都可以（托管的免费额度，或本地 ``docker run postgres``）。
对于异步 Web 服务器，还有 API 相同的 ``AsyncPostgresSaver``。
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from research_agent.config import settings


@contextmanager
def sqlite_saver(db_path: str | None = None) -> Iterator[SqliteSaver]:
    """产出一个绑定 ``db_path`` 的、可直接使用的 ``SqliteSaver``。

    请作为上下文管理器使用，使底层 SQLite 连接在工作期间保持打开、
    退出时干净关闭：

        with sqlite_saver() as saver:
            app = build_graph(checkpointer=saver)
            ...

    ``db_path`` 默认取 ``settings.checkpoint_db_path``。连接归 ``with``
    块所有——这正是长驻 Web 服务器（Stage 10）在整个生命周期内保持
    该块打开、而非按请求开关的原因。

    注意：``SqliteSaver.from_conn_string`` 会在首次使用时惰性创建检查点
    表，因此这里无需单独调用 ``setup()``。
    """
    path = db_path or settings.checkpoint_db_path
    with SqliteSaver.from_conn_string(path) as saver:
        yield saver
