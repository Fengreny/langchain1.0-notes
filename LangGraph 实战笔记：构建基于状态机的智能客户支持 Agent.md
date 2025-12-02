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
