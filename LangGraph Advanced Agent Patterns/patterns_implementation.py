import os
import getpass
import operator
from typing import Annotated, List, Literal, TypedDict, Union

# --- 依赖库检查 ---
# 请确保安装了以下库：
# pip install langchain_core langchain-anthropic langgraph pydantic

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Send
from pydantic import BaseModel, Field

# --- 1. 环境配置 (Setup) ---
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# 设置 API Key (这里假设你已经设置了环境变量，或者在运行时输入)
# _set_env("ANTHROPIC_API_KEY")

# 初始化 LLM (使用支持 Tool Calling 的模型)
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

print("=== 环境初始化完成 ===")

# ==========================================
# 模式一：提示词链 (Prompt Chaining)
# ==========================================
print("\n--- 模式 1: Prompt Chaining ---")

class ChainingState(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

def generate_joke(state: ChainingState):
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

def check_punchline(state: ChainingState):
    # 简单的门控机制：检查是否有标点符号暗示笑点
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"

def improve_joke(state: ChainingState):
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}

def polish_joke(state: ChainingState):
    # 如果有改进版，就润色改进版；否则润色原版（逻辑可根据需求调整）
    source_joke = state.get("improved_joke", state["joke"])
    msg = llm.invoke(f"Add a surprising twist to this joke: {source_joke}")
    return {"final_joke": msg.content}

chain_workflow = StateGraph(ChainingState)
chain_workflow.add_node("generate_joke", generate_joke)
chain_workflow.add_node("improve_joke", improve_joke)
chain_workflow.add_node("polish_joke", polish_joke)

chain_workflow.add_edge(START, "generate_joke")
chain_workflow.add_conditional_edges(
    "generate_joke",
    check_punchline,
    {"Fail": "improve_joke", "Pass": END} # 如果通过检查直接结束，否则去优化
)
chain_workflow.add_edge("improve_joke", "polish_joke")
chain_workflow.add_edge("polish_joke", END)

chain_app = chain_workflow.compile()
# 使用示例: chain_app.invoke({"topic": "cats"})


# ==========================================
# 模式二：并行化 (Parallelization)
# ==========================================
print("\n--- 模式 2: Parallelization ---")

class ParallelState(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

def call_llm_joke(state: ParallelState):
    return {"joke": llm.invoke(f"Write a joke about {state['topic']}").content}

def call_llm_story(state: ParallelState):
    return {"story": llm.invoke(f"Write a very short story about {state['topic']}").content}

def call_llm_poem(state: ParallelState):
    return {"poem": llm.invoke(f"Write a haiku about {state['topic']}").content}

def aggregator(state: ParallelState):
    combined = (
        f"Results for {state['topic']}:\n"
        f"JOKE: {state['joke']}\n"
        f"STORY: {state['story']}\n"
        f"POEM: {state['poem']}"
    )
    return {"combined_output": combined}

parallel_workflow = StateGraph(ParallelState)
parallel_workflow.add_node("call_llm_joke", call_llm_joke)
parallel_workflow.add_node("call_llm_story", call_llm_story)
parallel_workflow.add_node("call_llm_poem", call_llm_poem)
parallel_workflow.add_node("aggregator", aggregator)

# 并行分支
parallel_workflow.add_edge(START, "call_llm_joke")
parallel_workflow.add_edge(START, "call_llm_story")
parallel_workflow.add_edge(START, "call_llm_poem")
# 汇聚
parallel_workflow.add_edge("call_llm_joke", "aggregator")
parallel_workflow.add_edge("call_llm_story", "aggregator")
parallel_workflow.add_edge("call_llm_poem", "aggregator")
parallel_workflow.add_edge("aggregator", END)

parallel_app = parallel_workflow.compile()


# ==========================================
# 模式三：路由 (Routing)
# ==========================================
print("\n--- 模式 3: Routing ---")

class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="The next step in the routing process"
    )

router_llm = llm.with_structured_output(Route)

class RoutingState(TypedDict):
    input: str
    decision: str
    output: str

def llm_call_router(state: RoutingState):
    decision = router_llm.invoke([
        SystemMessage(content="Route the input to story, joke, or poem based on user request."),
        HumanMessage(content=state["input"])
    ])
    return {"decision": decision.step}

def route_decision(state: RoutingState):
    if state["decision"] == "story": return "write_story"
    elif state["decision"] == "joke": return "write_joke"
    elif state["decision"] == "poem": return "write_poem"

def write_story(state: RoutingState):
    return {"output": llm.invoke(f"Write a story about {state['input']}").content}
def write_joke(state: RoutingState):
    return {"output": llm.invoke(f"Write a joke about {state['input']}").content}
def write_poem(state: RoutingState):
    return {"output": llm.invoke(f"Write a poem about {state['input']}").content}

routing_workflow = StateGraph(RoutingState)
routing_workflow.add_node("llm_call_router", llm_call_router)
routing_workflow.add_node("write_story", write_story)
routing_workflow.add_node("write_joke", write_joke)
routing_workflow.add_node("write_poem", write_poem)

routing_workflow.add_edge(START, "llm_call_router")
routing_workflow.add_conditional_edges("llm_call_router", route_decision)
routing_workflow.add_edge("write_story", END)
routing_workflow.add_edge("write_joke", END)
routing_workflow.add_edge("write_poem", END)

routing_app = routing_workflow.compile()


# ==========================================
# 模式四：编排者-工作者 (Orchestrator-Worker)
# ==========================================
print("\n--- 模式 4: Orchestrator-Worker ---")

class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Brief overview of the main topics.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.")

planner = llm.with_structured_output(Sections)

class OrchState(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add] # 关键：合并列表
    final_report: str

class WorkerState(TypedDict):
    section: Section

def orchestrator(state: OrchState):
    report_sections = planner.invoke([
        SystemMessage(content="Generate a plan for the report."),
        HumanMessage(content=f"Here is the report topic: {state['topic']}")
    ])
    return {"sections": report_sections.sections}

def worker_node(state: WorkerState):
    section_content = llm.invoke([
        SystemMessage(content="Write a report section following the provided name and description."),
        HumanMessage(content=f"Section Name: {state['section'].name}\nDescription: {state['section'].description}")
    ])
    return {"completed_sections": [f"## {state['section'].name}\n\n{section_content.content}"]}

def synthesizer(state: OrchState):
    return {"final_report": "\n\n---\n\n".join(state["completed_sections"])}

def assign_workers(state: OrchState):
    return [Send("worker_node", {"section": s}) for s in state["sections"]]

orch_workflow = StateGraph(OrchState)
orch_workflow.add_node("orchestrator", orchestrator)
orch_workflow.add_node("worker_node", worker_node)
orch_workflow.add_node("synthesizer", synthesizer)

orch_workflow.add_edge(START, "orchestrator")
orch_workflow.add_conditional_edges("orchestrator", assign_workers, ["worker_node"])
orch_workflow.add_edge("worker_node", "synthesizer")
orch_workflow.add_edge("synthesizer", END)

orch_app = orch_workflow.compile()


# ==========================================
# 模式五：评估者-优化者 (Evaluator-Optimizer)
# ==========================================
print("\n--- 模式 5: Evaluator-Optimizer ---")

class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="Decide if the joke is funny or not.")
    feedback: str = Field(description="If not funny, provide feedback.")

evaluator_llm = llm.with_structured_output(Feedback)

class OptState(TypedDict):
    topic: str
    joke: str
    feedback: str
    funny_or_not: str

def generator_node(state: OptState):
    prompt = f"Write a joke about {state['topic']}"
    if state.get("feedback"):
        prompt += f" but take into account the feedback: {state['feedback']}"
    msg = llm.invoke(prompt)
    return {"joke": msg.content}

def evaluator_node(state: OptState):
    grade = evaluator_llm.invoke(f"Grade the joke: {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

def route_optimization(state: OptState):
    if state["funny_or_not"] == "funny":
        return END
    return "generator_node" # 循环

opt_workflow = StateGraph(OptState)
opt_workflow.add_node("generator_node", generator_node)
opt_workflow.add_node("evaluator_node", evaluator_node)

opt_workflow.add_edge(START, "generator_node")
opt_workflow.add_edge("generator_node", "evaluator_node")
opt_workflow.add_conditional_edges("evaluator_node", route_optimization)

opt_app = opt_workflow.compile()


# ==========================================
# 模式六：自主智能体 (Autonomous Agent)
# ==========================================
print("\n--- 模式 6: Autonomous Agent ---")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

tools = [multiply, add]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

def agent_llm_node(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def tool_node(state: MessagesState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

agent_workflow = StateGraph(MessagesState)
agent_workflow.add_node("agent_llm_node", agent_llm_node)
agent_workflow.add_node("tool_node", tool_node)

agent_workflow.add_edge(START, "agent_llm_node")
agent_workflow.add_conditional_edges("agent_llm_node", should_continue, ["tool_node", END])
agent_workflow.add_edge("tool_node", "agent_llm_node")

agent_app = agent_workflow.compile()

print("=== 所有模式定义完成，可直接调用对应的 app (如 agent_app.invoke(...)) ===")
