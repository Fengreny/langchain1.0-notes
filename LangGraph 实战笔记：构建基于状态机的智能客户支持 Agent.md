基于langgraph的理解是 需要把这个系统 想象成一个状态定义的机器。
使用 LangGraph 构建智能体时，首先要将其分解为称为节点的离散步骤。然后，描述每个节点的不同决策和状态转换。最后，通过一个共享状态将节点连接起来，每个节点都可以读取和写入该状态。

节点的步骤比较好理解就使用发邮件举例子：

假设你需要构建一个用于处理客户支持邮件的人工智能代理。

The agent should:

- Read incoming customer emails  阅读 收入 客户 的 邮件
- Classify them by urgency and topic 通过政策和主题来分类他们
- Search relevant documentation to answer questions  搜索相关的文档来回答问题
- Draft appropriate responses   草稿回答
- Escalate complex issues to human agents 将复杂问题上升到agent
- Schedule follow-ups when needed  需要时安排

Example scenarios to handle:

1. Simple product question: "How do I reset my password?"   简单产品问题："我怎么重置我的密码？"
2. Bug report: "The export feature crashes when I select PDF format" bug 问题 导出功能在我选择PDF格式时崩溃”
3. Urgent billing issue: "I was charged twice for my subscription!"     紧急账单问题："我被重复扣费了两次！"
4. Feature request: "Can you add dark mode to the mobile app?"   功能请求："你能在移动应用中添加暗黑模式吗？"
5. Complex technical issue: "Our API integration fails intermittently with 504 errors" 复杂技术问题："我们的API集成间歇性地出现504错误。"


要在 LangGraph 中实现代理，通常需要遵循相同的五个步骤。


第一步：将你的工作流程分解成一个个独立的步骤。

首先，确定流程中的各个步骤。每个步骤都将成为一个节点（一个执行特定操作的函数）。然后，绘制这些步骤之间的连接图。

Read Email提取并解析电子邮件内容
Classify Intent使用 LLM 对紧急程度和主题进行分类，然后路由到相应的行动。
Doc Search查询知识库以获取相关信息
Bug Track在跟踪系统中创建或更新问题
Draft Reply：生成适当的回应
Human Review：上报人工处理或审批。
Send Reply：发送邮件回复


=============================================================================================================================================
步骤二：明确每个步骤需要做什么

对于图中的每个节点，确定它代表什么类型的操作以及它需要什么上下文才能正常工作。

LLM步骤 （理解、分析、生成文本或做出推理决策时）

        当某个步骤需要理解、分析、生成文本或做出推理决策时：
        意图分类
        
        静态上下文（提示）：分类类别、紧急程度定义、响应格式
        动态上下文（来自状态）：电子邮件内容、发件人信息
        预期结果：确定路由的结构化分类
        草拟回复
        
        静态背景（提示）：语气准则、公司政策、回复模板
        动态上下文（来自状态）：分类结果、搜索结果、客户历史记录
        预期结果：可供审核的专业电子邮件回复
​



数据步骤（需要从外部来源检索信息时）

        文档搜索
        
        参数：根据意图和主题构建的查询
        重试策略：是，采用指数退避策略来应对瞬态故障。
        缓存：可以缓存常用查询以减少 API 调用次数。
        
        客户历史记录查询
        
        参数：客户电子邮件或州 ID
        重试策略：是，但如果无法获取基本信息，则回退到基本信息。
        缓存：是的，采用生存时间机制来平衡数据新鲜度和性能。


action 步骤

        当某个步骤需要执行外部操作时：
        回复
        
        节点执行时间：审批通过后（人工或自动审批）。
        重试策略：是，针对网络问题采用指数退避策略。
        不应缓存：每次发送都是一个独立的操作。
        
        缺陷跟踪
        
        何时执行节点：当 intent 为“bug”时始终执行
        重试策略：是的，这对于避免丢失错误报告至关重要。
        返回值：响应中包含的工单 ID

用户输入步骤

        当某个步骤需要人工干预时：
        
        人工审核节点
        
        决策背景：原始邮件、回复草稿、紧急程度、分类
        预期输入格式：批准布尔值，以及可选的编辑回复
        触发条件：高度紧急、问题复杂或存在质量问题

======================================================================================================

步骤 3：设计自己定义的状态

状态是智能体中所有节点均可访问的共享内存。您可以将其想象成智能体用来记录其在处理过程中学习和决策的笔记本。
​
哪些东西属于state管辖范围？
针对每条数据，请问自己以下问题：
        包含在state内：它是否需要跨步骤保持？如果需要，则将其置于某种状态中。
        不要储存：能否从其他数据中推导出它？如果可以，则在需要时计算，而不是将其存储在状态中。
        
        对于我们的邮件代理，我们需要跟踪：
        原始邮件和发件人信息（之后无法恢复）
        分类结果（多个后续/下游节点需要）
        搜索结果和客户数据（重新获取成本高昂）
        答复草稿（需要保留到审查阶段）
        执行元数据（用于调试和恢复）
        ​
        
        一个关键原则：你的状态应该存储原始数据，而不是格式化的文本。需要时，在节点内部添加格式化提示。
        
        这种分离意味着：
        不同的节点可以根据自身需求对相同的数据进行不同的格式化。
        您无需修改​​状态架构即可更改提示模板。
        调试更清晰——您可以清楚地看到每个节点接收到了哪些数据。
        你的代理可以在不破坏现有状态的情况下进化
        
=========================================================================================================

骤 4：构建节点
现在我们将每个步骤实现为一个函数。LangGraph 中的一个节点就是一个 Python 函数，它接收当前状态并返回更新后的状态。

妥善处理错误
不同的错误需要不同的处理策略：
1. 网络问题、速率限制 ：
   
   添加重试策略，以自动重试网络问题和速率限制问题：
        from langgraph.types import RetryPolicy
        
        workflow.add_node(
            "search_documentation",
            search_documentation,
            retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
        )
   
2. 将错误存储在状态中并循环返回，以便 LLM 可以查看哪里出了问题并重试：
        from langgraph.types import Command
        
        
        def execute_tool(state: State) -> Command[Literal["agent", "execute_tool"]]:
            try:
                result = run_tool(state['tool_call'])
                return Command(update={"tool_result": result}, goto="agent")
            except ToolError as e:
                # Let the LLM see what went wrong and try again
                return Command(
                    update={"tool_result": f"Tool error: {str(e)}"},
                    goto="agent"
                )
3. 用户可修正的错误（信息缺失、说明不清晰），必要时暂停并从用户处收集信息（例如帐户 ID、订单号或说明信息）：
        from langgraph.types import Command
        
        
        def lookup_customer_history(state: State) -> Command[Literal["draft_response"]]:
            if not state.get('customer_id'):
                user_input = interrupt({
                    "message": "Customer ID needed",
                    "request": "Please provide the customer's account ID to look up their subscription history"
                })
                return Command(
                    update={"customer_id": user_input['customer_id']},
                    goto="lookup_customer_history"
                )
            # Now proceed with the lookup
            customer_data = fetch_customer_history(state['customer_id'])
            return Command(update={"customer_history": customer_data}, goto="draft_response")
4.  意外，不要处理你无法解决的问题：
        def send_reply(state: EmailAgentState):
            try:
                email_service.send(state["draft_response"])
            except Exception:
                raise  # Surface unexpected errors


======================================================================================================================
实现一个邮件agent：


1. 读取并分类节点：
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4")

def read_email(state: EmailAgentState) -> dict:
    """Extract and parse email content"""
    # In production, this would connect to your email service
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """Use LLM to classify email intent and urgency, then route accordingly"""

    # Create structured LLM that returns EmailClassification dict
    structured_llm = llm.with_structured_output(EmailClassification)

    # Format the prompt on-demand, not stored in state
    classification_prompt = f"""
    Analyze this customer email and classify it:

    Email: {state['email_content']}
    From: {state['sender_email']}

    Provide classification including intent, urgency, topic, and summary.
    """

    # Get structured response directly as dict
    classification = structured_llm.invoke(classification_prompt)

    # Determine next node based on classification
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    # Store classification as a single dict in state
    return Command(
        update={"classification": classification},
        goto=goto
    )

2. 搜索和跟踪节点
def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Search knowledge base for relevant information"""

    # Build search query from classification
    classification = state.get('classification', {})
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

    try:
        # Implement your search logic here
        # Store raw search results, not formatted text
        search_results = [
            "Reset password via Settings > Security > Change Password",
            "Password must be at least 12 characters",
            "Include uppercase, lowercase, numbers, and symbols"
        ]
    except SearchAPIError as e:
        # For recoverable search errors, store error and continue
        search_results = [f"Search temporarily unavailable: {str(e)}"]

    return Command(
        update={"search_results": search_results},  # Store raw results or error
        goto="draft_response"
    )

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Create or update bug tracking ticket"""

    # Create ticket in your bug tracking system
    ticket_id = "BUG-12345"  # Would be created via API

    return Command(
        update={
            "search_results": [f"Bug ticket {ticket_id} created"],
            "current_step": "bug_tracked"
        },
        goto="draft_response"
    )

3. 响应节点
def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Generate response using context and route based on quality"""

    classification = state.get('classification', {})

    # Format context from raw state data on-demand
    context_sections = []

    if state.get('search_results'):
        # Format search results for the prompt
        formatted_docs = "\n".join([f"- {doc}" for doc in state['search_results']])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")

    if state.get('customer_history'):
        # Format customer data for the prompt
        context_sections.append(f"Customer tier: {state['customer_history'].get('tier', 'standard')}")

    # Build the prompt with formatted context
    draft_prompt = f"""
    Draft a response to this customer email:
    {state['email_content']}

    Email intent: {classification.get('intent', 'unknown')}
    Urgency level: {classification.get('urgency', 'medium')}

    {chr(10).join(context_sections)}

    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant
    """

    response = llm.invoke(draft_prompt)

    # Determine if human review needed based on urgency and intent
    needs_review = (
        classification.get('urgency') in ['high', 'critical'] or
        classification.get('intent') == 'complex'
    )

    # Route to appropriate next node
    goto = "human_review" if needs_review else "send_reply"

    return Command(
        update={"draft_response": response.content},  # Store only the raw response
        goto=goto
    )

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""

    classification = state.get('classification', {})

    # interrupt() must come first - any code before it will re-run on resume
    human_decision = interrupt({
        "email_id": state.get('email_id',''),
        "original_email": state.get('email_content',''),
        "draft_response": state.get('draft_response',''),
        "urgency": classification.get('urgency'),
        "intent": classification.get('intent'),
        "action": "Please review and approve/edit this response"
    })

    # Now process the human's decision
    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response',''))},
            goto="send_reply"
        )
    else:
        # Rejection means human will handle directly
        return Command(update={}, goto=END)

def send_reply(state: EmailAgentState) -> dict:
    """Send the email response"""
    # Integrate with email service
    print(f"Sending reply: {state['draft_response'][:100]}...")
    return {}


=======================================================================================================================
第五步：将它们连接起来


from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

# Create the graph
workflow = StateGraph(EmailAgentState)

# Add nodes with appropriate error handling
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)

# Add retry policy for nodes that might have transient failures
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3)
)
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

# Add only the essential edges
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# Compile with checkpointer for persistence, in case run graph with Local_Server --> Please compile without checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


=============================================================================================================================

测试
# Test with an urgent billing issue
initial_state = {
    "email_content": "I was charged twice for my subscription! This is urgent!",
    "sender_email": "customer@example.com",
    "email_id": "email_123",
    "messages": []
}

# Run with a thread_id for persistence
config = {"configurable": {"thread_id": "customer_123"}}
result = app.invoke(initial_state, config)
# The graph will pause at human_review
print(f"Draft ready for review: {result['draft_response'][:100]}...")

# When ready, provide human input to resume
from langgraph.types import Command

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund..."
    }
)

# Resume execution
final_result = app.invoke(human_response, config)
print(f"Email sent successfully!")
