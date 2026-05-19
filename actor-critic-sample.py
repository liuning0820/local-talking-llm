import operator
from typing import Annotated, TypedDict, List
from openai import OpenAI
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
LLAMA_SERVER_MODEL = os.getenv("LLAMA_SERVER_MODEL", "DeepSeek-R1-Distill-Qwen-1.5B")
LLAMA_SERVER_BASE_URL = os.getenv("LLAMA_SERVER_BASE_URL", "http://localhost:8080/v1")


client = OpenAI(
    base_url=LLAMA_SERVER_BASE_URL,
    api_key='sk-no-key-required', # llama.cpp server doesn't require a real API key
)


def _get_choice_content(response):
    """
    Safely extract text content from an OpenAI-like response object.
    Supports choice objects as dicts or objects with attributes.
    """
    if not response or not getattr(response, "choices", None):
        return ""
    choice = response.choices[0]
    # dict-like
    if isinstance(choice, dict):
        if "message" in choice and isinstance(choice["message"], dict):
            return choice["message"].get("content", "") or ""
        return choice.get("content", "") or ""
    # object-like
    msg = getattr(choice, "message", None)
    if msg is not None:
        return getattr(msg, "content", "") or ""
    return getattr(choice, "content", "") or ""

# 1. 定义工作流的状态 (State)
# 状态会在 Actor 和 Critic 之间传递，记录任务的进度
class AgentState(TypedDict):
    task: str                # 用户输入的原始任务
    draft: str               # Actor 生成的草稿/推理过程
    critique: str            # Critic 给出的反馈/批评
    revision_count: int      # 当前修正的次数
    final_output: str        # 最终满意的输出

# 2. 定义 Actor 节点 (负责生成与修正)
def actor_node(state: AgentState):
    task = state["task"]
    critique = state.get("critique", "")
    
    # 如果有 Critic 的反馈，Actor 需要基于反馈进行修正
    if critique:
        system_prompt = f"""你是一个专业的 AI 助手。你之前生成的内容收到了以下修改意见：
        【修改意见】: {critique}
        请根据意见，重新优化你的回答。"""
        user_prompt = f"任务目标：{task}"
    else:
        # 第一次生成，直接根据任务执行
        system_prompt = "你是一个专业的 AI 助手，请根据用户的要求进行详细的推理和回答。"
        user_prompt = task

    response = client.chat.completions.create(
        model=LLAMA_SERVER_MODEL, # 或任意强大的 LLM 作为 Actor
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    draft_text = _get_choice_content(response)
    return {"draft": draft_text, "revision_count": state.get("revision_count", 0) + 1}

# 3. 定义 Critic 节点 (负责评估与挑错)
def critic_node(state: AgentState):
    draft = state["draft"]
    task = state["task"]
    
    system_prompt = """你是一个极其严格的质量评判专家（Critic）。
    你的任务是审查 AI 生成的草稿，找出其中的逻辑漏洞、事实错误、格式问题或未尽之处。
    如果草稿已经非常完美，请只回复 "PASS"。
    如果有问题，请给出具体的、可操作的修改建议（自然语言反馈）。"""
    
    user_prompt = f"""请审查以下针对任务【{task}】生成的草稿：
    ---
    {draft}
    ---"""

    response = client.chat.completions.create(
        model=LLAMA_SERVER_MODEL, # 可以用一个逻辑强或更挑剔的模型作为 Critic
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    critique_text = _get_choice_content(response)
    return {"critique": critique_text}

# 4. 定义路由逻辑 (决定是继续修正还是结束)
def route(state: AgentState):
    # 如果 Critic 说 "PASS"，或者修正次数超过 3 次（防止死循环），则结束
    if "PASS" in state["critique"].upper() or state["revision_count"] > 3:
        return "end"
    else:
        return "revise"

# 5. 构建与编译图 (Graph)
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("actor", actor_node)
workflow.add_node("critic", critic_node)

# 设置入口点（从 Actor 开始）
workflow.set_entry_point("actor")

# 添加边：Actor -> Critic
workflow.add_edge("actor", "critic")

# 添加条件边：Critic -> (Actor 或 END)
workflow.add_conditional_edges(
    "critic",
    route,
    {"revise": "actor", "end": END}
)

# 编译工作流
app = workflow.compile()

# 6. 运行工作流
def run_actor_critic(task_query: str):
    initial_state = {"task": task_query, "revision_count": 0}
    print(f"🚀 开始处理任务: {task_query}\n")
    
    # 流式输出每一步的状态
    for output in app.stream(initial_state):
        if "actor" in output:
            print(f"🤖 [Actor 第 {output['actor']['revision_count']} 版草稿]:\n{output['actor']['draft'][:200]}...\n")
        if "critic" in output:
            print(f"⚖️ [Critic 反馈]: {output['critic']['critique']}\n" + "-"*50 + "\n")
            
    print("✅ 任务处理完成！")

# 测试运行
if __name__ == "__main__":
    task = "请用 Python 写一个带有详细注释的快速排序算法，并解释其时间复杂度。"
    run_actor_critic(task)