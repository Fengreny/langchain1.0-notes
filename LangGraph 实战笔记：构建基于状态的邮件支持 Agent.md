# 用图的思维: 构建 AI Agent 的核心思维

> 基于 LangChain 官方文档 "Thinking in LangGraph" 的深度笔记。
> 学习如何将业务流程转化为基于图（Graph）的智能体系统。

## 核心理念

在使用 LangGraph 构建 Agent 时，不要把它仅仅看作代码的堆砌，而应该将其想象成一个**状态机**。

构建过程遵循以下三个核心概念：
1.  **Nodes (节点)**：将工作流拆解为离散的步骤（函数）。
2.  **Edges & Decisions (边与决策)**：定义节点之间的流转逻辑。
3.  **State (状态)**：一个共享的“笔记本”，所有节点都可以从中读取数据或写入更新。

---

## 实战案例：客户支持邮件 Agent

假设我们需要构建一个自动处理客户邮件的 Agent，需求如下：

*   **输入**：读取客户邮件。
*   **处理**：
    *   按紧急程度和主题分类。
    *   搜索文档回答问题。
    *   处理 Bug 报告。
    *   草拟回复。
*   **人工介入**：复杂问题或高风险操作需人工审核。
*   **输出**：发送回复。

### 5步构建法 (The 5-Step Process)

### 第一步：绘制工作流 (Map out workflow)

首先，将连续的业务流程拆解为独立的**节点**。

*   **Read Email**: 提取并解析邮件内容。
*   **Classify Intent**: 使用 LLM 判断意图（咨询、Bug、账单等）和紧急程度。
*   **Doc Search**: 查询知识库（针对咨询类）。
*   **Bug Track**: 提交工单到追踪系统（针对 Bug 类）。
*   **Draft Reply**: 根据上下文生成回复草稿。
*   **Human Review**: 人工审核（针对高危/复杂情况）。
*   **Send Reply**: 发送最终邮件。

### 第二步：明确节点功能 (Identify step needs)

分析每个节点属于哪种类型，以及它需要什么上下文。

| 节点类型 | 典型操作 | 示例节点 | 关键点 |
| :--- | :--- | :--- | :--- |
| **LLM 步骤** | 理解、分析、生成 | `Classify Intent`<br>`Draft Reply` | **输入**：Prompt + 状态数据<br>**输出**：结构化决策或文本 |
| **数据步骤** | 外部检索 | `Doc Search`<br>`Customer History` | **策略**：需考虑缓存（Caching）和重试（Retry） |
| **Action 步骤** | 执行外部动作 | `Send Reply`<br>`Bug Track` | **策略**：通常不缓存，需严格的重试策略 |
| **用户输入** | 人工干预 | `Human Review` | **机制**：使用 `interrupt` 暂停执行，等待用户反馈 |

### 第三步：设计状态 (Design State)

**State** 是所有节点的共享内存。

> **关键原则**：State 应该存储**原始数据 (Raw Data)**，而不是格式化后的 Prompt。
> *   **原因**：不同的节点可能需要以不同方式使用同一份数据。格式化（Formatting）应在节点内部进行。

我们需要在 Python 中定义这个结构（通常使用 `TypedDict`）：

```python
from typing import TypedDict, Literal, List, Optional

# 定义分类结果的结构
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

# 定义整个 Agent 的共享状态
class EmailAgentState(TypedDict):
    # 1. 原始输入数据
    email_content: str
    sender_email: str
    email_id: str

    # 2. 节点的处理结果
    classification: Optional[EmailClassification] # 分类结果
    search_results: Optional[List[str]]           # 搜索到的原始文档块
    customer_history: Optional[dict]              # 客户信息

    # 3. 生成的内容
    draft_response: Optional[str]                 # 回复草稿
    messages: List[str]                           # 消息历史
```

### 第四步：构建节点 (Build Nodes)

节点本质上就是 Python 函数：`Input(State) -> Output(Update) + Routing`。

在此步骤中，我们需要处理四种类型的错误：
1.  **瞬态错误 (网络/API)**：使用 `RetryPolicy` 自动重试。
2.  **LLM 错误 (解析失败)**：捕获错误并存入 State，让 LLM 重试。
3.  **用户可修正错误 (缺信息)**：使用 `interrupt` 暂停并请求用户输入。
4.  **意外错误**：抛出异常，人工排查。

#### 核心代码实现

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage

# 初始化 LLM
llm = ChatOpenAI(model="gpt-4")

# --- 1. 读取与分类节点 ---

def read_email(state: EmailAgentState) -> dict:
    """读取邮件（模拟）"""
    print(f"📥 Reading email from {state['sender_email']}")
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """使用 LLM 分类意图并路由"""
    
    # 使用结构化输出
    structured_llm = llm.with_structured_output(EmailClassification)

    # 在节点内动态构建 Prompt
    prompt = f"""
    Analyze this email:
    Content: {state['email_content']}
    From: {state['sender_email']}
    Provide classification (intent, urgency, topic, summary).
    """
    
    classification = structured_llm.invoke(prompt)

    # 路由逻辑：根据分类结果决定下一步
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    # 返回 Command：更新状态 + 跳转
    return Command(
        update={"classification": classification},
        goto=goto
    )

# --- 2. 工具与数据节点 ---

def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """搜索文档"""
    cls = state.get('classification', {})
    query = f"{cls.get('intent')} {cls.get('topic')}"
    
    # 模拟搜索结果
    results = [
        "Reset password via Settings > Security",
        "Password requirements: 12+ chars"
    ]
    
    return Command(
        update={"search_results": results},
        goto="draft_response"
    )

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """提交 Bug 工单"""
    ticket_id = "BUG-12345"
    return Command(
        update={
            "search_results": [f"Bug ticket {ticket_id} created"], # 复用 search_results 字段存储上下文
        },
        goto="draft_response"
    )

# --- 3. 生成与响应节点 ---

def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """生成回复草稿"""
    cls = state.get('classification', {})
    
    # 组装上下文
    context = []
    if state.get('search_results'):
        context.append(f"Docs: {state['search_results']}")
    
    prompt = f"""
    Draft a response to: {state['email_content']}
    Intent: {cls.get('intent')}
    Context: {context}
    """
    
    response = llm.invoke(prompt)
    
    # 再次检查是否需要人工审核
    needs_review = cls.get('urgency') in ['high', 'critical']
    goto = "human_review" if needs_review else "send_reply"
    
    return Command(
        update={"draft_response": response.content},
        goto=goto
    )

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """人工审核节点 (Human-in-the-loop)"""
    
    # 1. 中断执行，等待用户输入
    # interrupt 之前的所有代码在恢复时会重跑，所以通常把 interrupt 放在最前面
    feedback = interrupt({
        "task": "review_draft",
        "email_content": state['email_content'],
        "draft": state['draft_response']
    })
    
    # 2. 恢复执行后，处理用户反馈
    if feedback.get("approved"):
        final_response = feedback.get("edited_response", state['draft_response'])
        return Command(
            update={"draft_response": final_response},
            goto="send_reply"
        )
    else:
        # 如果被拒绝，结束流程（或跳转到其他处理节点）
        print("🚫 Draft rejected by human.")
        return Command(update={}, goto=END)

def send_reply(state: EmailAgentState):
    """发送邮件"""
    print(f"🚀 Sending Email: {state['draft_response']}")
    return {}
```

### 第五步：连接图谱 (Wire it together)

最后，使用 `StateGraph` 将节点组装起来，并配置持久化存储（Checkpointer）以支持中断恢复。

```python
from langgraph.checkpoint.memory import MemorySaver

# 1. 创建图
workflow = StateGraph(EmailAgentState)

# 2. 添加节点
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)
# 为易失败的节点添加重试策略
workflow.add_node("search_documentation", search_documentation, retry_policy=RetryPolicy(max_attempts=3))
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

# 3. 添加起始边
workflow.add_edge(START, "read_email")
# 注意：其他边已在节点内部通过 Command(goto=...) 动态定义，无需在此硬编码

# 4. 编译图（启用 Checkpointer）
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

### 测试运行

模拟一个需要人工审核的紧急场景。

```python
# 初始状态
initial_state = {
    "email_content": "I was charged twice! Urgent!",
    "sender_email": "vip@example.com",
    "email_id": "mail_001",
    "messages": []
}

# 配置线程 ID (用于持久化记忆)
config = {"configurable": {"thread_id": "ticket_001"}}

print("--- 第一次运行 (直到中断) ---")
# 运行图，它会在 human_review 处暂停
for event in app.stream(initial_state, config):
    pass 

# 此时，我们可以检查状态
snapshot = app.get_state(config)
print(f"\n⏸️ Paused at: {snapshot.next}")
print(f"Draft content: {snapshot.values['draft_response']}")

print("\n--- 提供人工反馈并恢复 ---")
# 提供反馈数据
human_feedback = Command(
    resume={
        "approved": True,
        "edited_response": "Sorry for the double charge. Refund processed."
    }
)

# 恢复执行
final_result = app.invoke(human_feedback, config)
```

## 总结

LangGraph 的核心在于**显式控制流**。通过将 Agent 拆解为节点，并利用 State 存储原始数据，我们能够构建出比纯 Prompt 工程更稳健、可调试、可扩展的 AI 应用。
