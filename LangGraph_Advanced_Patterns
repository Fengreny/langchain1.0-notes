# LangGraph Advanced Agent Patterns

## 📖 简介 (Introduction)

本项目基于 [LangGraph](https://github.com/langchain-ai/langgraph) 框架，系统性地复现并整理了 6 种核心的大语言模型（LLM）工作流与智能体（Agent）设计模式。

这些模式涵盖了从确定性的工作流控制到自主智能体的动态决策，旨在解决复杂任务分解、长文本处理及高质量数据生成等实际工程问题。

## 🚀 核心模式 (Core Patterns)

本项目实现了以下设计模式，每种模式都针对特定的应用场景进行了优化：

### 1. 提示词链 (Prompt Chaining)
*   **原理**：将任务分解为线性的步骤，上一步的输出作为下一步的输入。
*   **应用**：生成 -> 检查 -> 优化。适合逻辑固定的任务，如文案润色。

### 2. 并行化 (Parallelization)
*   **原理**：同时运行多个独立的子任务，最后聚合结果。
*   **应用**：多维度分析、多风格生成。能显著降低总延时。

### 3. 路由 (Routing)
*   **原理**：利用 LLM 的结构化输出（Structured Output）对输入进行分类，导向不同的处理路径。
*   **应用**：智能客服分流、意图识别系统。

### 4. 编排者-工作者 (Orchestrator-Worker)
*   **原理**：Orchestrator 负责规划和拆解任务，Worker 负责并行执行具体子任务，最后由 Synthesizer 汇总。
*   **应用**：**长文档处理**、代码库重构。此模式通过动态创建 Worker，有效突破了 Context Window 的限制。

### 5. 评估者-优化者 (Evaluator-Optimizer)
*   **原理**：引入反馈循环（Feedback Loop）。Generator 生成内容，Evaluator 评分并提供修改建议，直至满足标准。
*   **应用**：**高质量数据集生成**、代码自动修复。通过自我反思（Self-Reflection）机制提升输出质量。

### 6. 自主智能体 (Autonomous Agents)
*   **原理**：LLM 处于循环中心，自主决定是否调用工具（Tools）以及调用的顺序。
*   **应用**：开放性问题解决、ReAct 模式实现。

## 🛠️ 技术栈 (Tech Stack)

*   **Python 3.10+**
*   **LangGraph**: 用于构建有状态的、多角色的图结构应用。
*   **LangChain**: 提供基础的 Prompt 模板和 Tool 抽象。
*   **Anthropic API (Claude 3.5 Sonnet)**: 提供强大的指令遵循和代码生成能力。
*   **Pydantic**: 用于定义结构化输出的数据模型。

## 📂 文件结构 (File Structure)

```text
LangGraph_Advanced_Patterns/
├── README.md                  # 项目说明文档
└── patterns_implementation.py # 包含所有 6 种模式的完整实现代码
```

## 💡 使用说明 (Usage)

1.  安装依赖：
    ```bash
    pip install langchain_core langchain-anthropic langgraph pydantic
    ```

2.  设置环境变量（推荐在 `.env` 文件中配置）：
    ```bash
    export ANTHROPIC_API_KEY="your_api_key_here"
    ```

3.  运行代码：
    您可以直接运行 `patterns_implementation.py`，或者将其中的类和函数导入到您的项目中。代码中已包含每种模式的 `StateGraph` 构建逻辑。

## 📝 个人研学笔记

*   **Workflow vs Agent**: Workflow 提供了可控性和可靠性，适合生产环境中的关键路径；Agent 提供了灵活性，适合处理长尾问题。
*   **工程落地思考**: 在实际的 Multi-Agent 系统（如 Idea 3）中，我发现 **Evaluator-Optimizer** 模式对于清洗合成数据非常有效，而 **Orchestrator-Worker** 则是处理长上下文（Idea 4）的首选架构。

---
*Created by Jintao*
