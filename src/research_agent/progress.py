"""运行进度的终端输出。

所有节点通过 ``progress()`` 打印中间步骤，让操作员在长时间运行的
LLM 调用之间始终看得见图正在做什么。走 stderr 而非 stdout：最终报告
等"正经输出"走 stdout，进度日志与之分离，重定向/管道时互不污染。

关闭方式：``VERBOSE=false``（见 config.Settings.verbose）。
"""

from __future__ import annotations

import sys
import time

from research_agent.config import settings


def progress(message: str) -> None:
    """打印一行带时间戳的进度信息；settings.verbose 为 False 时静默。"""
    if not settings.verbose:
        return
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)
