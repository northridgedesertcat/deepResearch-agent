from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_planner: str = "openai:gpt-4o"
    model_researcher: str = "openai:gpt-4o-mini"
    model_synthesizer: str = "openai:gpt-4o"
    model_critic: str = "openai:gpt-4o-mini"
    model_judge: str = "openai:gpt-4o-mini"   # 供 LLM-as-judge 评估使用的廉价模型

    temperature: float = 0.0
    max_research_iterations: int = 2

    # LangChain 从模型获取结构化（Pydantic）输出的方式。
    #   "json_schema"      → OpenAI 原生结构化输出；约束最严格，但仅 OpenAI
    #                        本身支持（其他提供商会以 400 拒绝）
    #   "function_calling" → 基于工具调用；兼容 DeepSeek 及大多数 OpenAI 兼容
    #                        提供商 —— 使用 DeepSeek 时请设置此项
    #   "json_mode"        → response_format={"type": "json_object"}
    structured_output_method: str = "json_schema"

    # 运行可见性。为 True 时各节点把正在执行的操作打印到 stderr（带时间戳），
    # 交互式运行时可以看清图在做什么；纯 API/生产部署可设为 False 保持日志干净。
    verbose: bool = True

    # 评估门槛：智能体在 CI 中的平均分不得低于这些值。
    eval_min_faithfulness: float = 0.7
    eval_min_citation_accuracy: float = 0.8
    eval_min_coverage: float = 0.5
    eval_max_cases: int = 0   # 0 = 跑完整数据集；>0 则限制数量（降低 CI 成本）

    # ── 护栏与健壮性 ─────────────────────────────────────────────────
    # 输入护栏。分类器判断查询是否属于本领域、且可以安全执行。
    # `guardrail_domain` 会注入其提示词，因此无需改代码即可调整范围。
    guardrail_input_enabled: bool = True
    guardrail_domain: str = (
        "可基于公开网络来源研究的事实性问题"
        "（科技、科学、商业、历史、时事等类似领域）"
    )

    # 输出护栏。报告中被引用的来源至少要有这个比例确实来自已采集来源，
    # 否则退回 synthesizer 修复——最多修复 `max_output_repairs` 次，
    # 之后打标记并照常发布。
    guardrail_output_enabled: bool = True
    min_citation_coverage: float = 0.8
    max_output_repairs: int = 1

    # 预算护栏。每次运行的 LLM 调用硬上限。与 `max_research_iterations`
    # 结合，确保单个查询不会失控运行。
    max_llm_calls_per_run: int = 22

    # 健壮性。应用于所有 LLM 节点的重试策略，使瞬态失败（超时、429 限流）
    # 退避重试，而不是让整个运行崩溃。
    retry_max_attempts: int = 4
    retry_initial_interval: float = 1.0   # 秒；每次尝试翻倍（1、2、4、8）
    retry_backoff_factor: float = 2.0

    # ── 持久化与人在回路 ───────────────────────────────────
    # SQLite 检查点器写入持久化状态的位置。单个本地文件；零配置、零成本。
    # 生产环境请换用 PostgresSaver（见 research_agent/persistence.py）。
    # 相对路径以当前工作目录为基准解析，因此和 .env 一样要求从项目根目录运行。
    checkpoint_db_path: str = "checkpoints.db"

    # 人在回路审批门。为 True 时，图在生成第一份计划后（在昂贵的扇出之前）
    # 立即暂停，等待人工批准 / 编辑 / 拒绝子任务。默认关闭，使智能体在评估、
    # CI 和 API 默认路径上保持完全自主——交互式/运维使用时按次开启。
    # 注意：HITL 仅在有检查点器时生效；interrupt() 需要持久化状态才能暂停和恢复。
    hitl_approval_enabled: bool = False


settings = Settings()