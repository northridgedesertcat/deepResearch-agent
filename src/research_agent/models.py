from functools import lru_cache
from langchain.chat_models import init_chat_model
from research_agent.config import settings
from research_agent.progress import progress

_SPEC_BY_ROLE = {
    "planner": settings.model_planner,
    "researcher": settings.model_researcher,
    "synthesizer": settings.model_synthesizer,
    "critic": settings.model_critic,
    "judge": settings.model_judge,
}


@lru_cache(maxsize=None)
def get_model(role: str):
    # 按逻辑角色返回聊天模型，与具体提供商无关。
    spec = _SPEC_BY_ROLE[role]
    return init_chat_model(spec, temperature=settings.temperature)


def invoke_structured(role: str, schema, prompt: str):
    """调用该角色的模型并返回解析后的 ``schema`` 实例。

    OpenAI 兼容提供商（如 DeepSeek）偶尔会用普通文本回答，而不是预期的
    工具调用，LangChain 会将其解析为 ``None``——长提示词时更容易发生。
    在 temperature 0 下每次重试都会得到完全相同的答案，因此采用阶梯式升级：

      1. 配置的方法 + 配置的 temperature
      2. 配置的方法 + temperature 0.7  （不同的采样）
      3. json_mode + temperature 0.7    （完全不同的机制）

    节点级 RETRY_POLICY 在这里帮不上忙：它只覆盖网络错误，不覆盖
    ``None`` 结果。
    """
    spec = _SPEC_BY_ROLE[role]
    keys = ", ".join(schema.model_fields)
    attempts = (
        (settings.temperature, settings.structured_output_method, ""),
        (0.7, settings.structured_output_method, ""),
        (0.7, "json_mode",
         f"\n\n请输出一个仅包含以下键的 JSON 对象：{keys}。所有字符串值一律使用中文。"),
    )
    last: Exception | None = None
    for attempt, (temp, method, suffix) in enumerate(attempts, 1):
        model = init_chat_model(spec, temperature=temp).with_structured_output(schema, method=method)
        try:
            progress(f"… {role}：请求结构化输出（第 {attempt}/{len(attempts)} 档，temperature={temp}，method={method}）")
            result = model.invoke(prompt + suffix)
            if result is not None:
                return result
            progress(f"⚠ {role}：模型返回了空结果（temperature={temp}，method={method}）→ 升级到下一档重试")
        except Exception as exc:  # 校验/提供商错误 → 尝试下一档
            last = exc
            progress(f"⚠ {role}：第 {attempt} 档尝试失败（temperature={temp}，method={method}）：{exc} → 升级到下一档重试")
    raise ValueError(
        f"{role}：经过 {len(attempts)} 次升级尝试仍未获得结构化输出；最后一次错误：{last}"
    )