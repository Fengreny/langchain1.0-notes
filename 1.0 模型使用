安装 LangChain 及对应模型提供商的适配库：

```bash
# 安装核心库
pip install -U langchain langchain-core

# 根据需要安装模型提供商库
pip install -U langchain-openai    # OpenAI
pip install -U langchain-anthropic # Anthropic (Claude)
pip install -U langchain-google-genai # Google Gemini
```

---

## 1. 统一初始化 (Standard Initialization)

LangChain 推荐使用 `init_chat_model` 接口。它提供了一种**与提供商无关**的方式来加载模型，极大降低了在不同模型（如 GPT-4 和 Claude 3.5）之间切换的成本。

### 基础用法

```python
import os
from langchain.chat_models import init_chat_model

# 设置 API Key (建议通过环境变量管理)
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-..."

# 初始化 OpenAI 模型
model_gpt = init_chat_model("gpt-4o", model_provider="openai", temperature=0)

# 初始化 Anthropic 模型 (仅需更改参数，接口一致)
model_claude = init_chat_model("claude-3-5-sonnet-20240620", model_provider="anthropic", temperature=0)

# 测试调用
response = model_gpt.invoke("Hello, world!")
print(response.content)
```

### 常用参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `model` | `str` | 模型名称 (如 `gpt-4o`, `claude-3-5-sonnet`) |
| `model_provider` | `str` | 提供商标识 (如 `openai`, `anthropic`, `google_genai`) |
| `temperature` | `float` | 随机性 (0-1)，0 为最确定，1 为最发散 |
| `max_tokens` | `int` | 最大输出 Token 限制 |

---

## 2. 调用模式 (Invocation Patterns)

模型支持三种主要的交互方式，适用于不同的业务场景。

### 2.1 同步调用 (Invoke)
最基础的请求-响应模式。

```python
response = model.invoke("为什么鹦鹉会说话？")
print(f"回答: {response.content}")
# 返回类型: AIMessage
```

### 2.2 流式输出 (Stream)
**推荐用于 Chat UI**。逐步返回 Token，减少用户感知的延迟。

```python
print("正在生成: ", end="", flush=True)
for chunk in model.stream("给我讲一个关于编程的短笑话"):
    # chunk.content 包含当前生成的片段
    print(chunk.content, end="|", flush=True)
```

### 2.3 批量处理 (Batch)
并发处理多个独立请求，提高吞吐量。

```python
questions = [
    "1+1等于几？",
    "中国的首都是哪里？",
    "Python 是什么类型的语言？"
]

# 并发执行
responses = model.batch(questions)
for res in responses:
    print(res.content)
```

---

## 3. 工具调用 (Tool Calling)

工具调用是 **Agent** 的基础。模型可以“决定”调用外部函数（如搜索、数据库查询），并利用结果生成回答。

### 流程图解
1. **Bind**: 将工具定义绑定到模型。
2. **Invoke**: 模型返回 `tool_calls` 请求。
3. **Execute**: 代码执行工具逻辑。
4. **Response**: 将工具结果回传给模型。

### 代码示例

```python
from langchain.tools import tool

# 1. 定义工具 (使用 @tool 装饰器)
@tool
def get_weather(location: str) -> str:
    """查询指定地区的天气信息。"""
    # 模拟 API 调用
    if "北京" in location:
        return "北京今天晴朗，25度。"
    return f"{location} 天气未知。"

# 2. 绑定工具到模型
model_with_tools = model.bind_tools([get_weather])

# 3. 模型决策
query = "北京今天天气怎么样？"
response = model_with_tools.invoke(query)

# 4. 检查是否触发工具调用
if response.tool_calls:
    print(f"模型请求调用工具: {response.tool_calls}")
    # 输出示例: [{'name': 'get_weather', 'args': {'location': '北京'}, 'id': '...'}]
    
    # 注意: 在 Agent 框架中，执行步骤会自动处理。
    # 在 Standalone 模式下，你需要手动解析 tool_calls 并执行函数。
```

---

## 4. 结构化输出 (Structured Output)

在全栈开发中，通常需要模型返回 **JSON** 或 **对象** 而非纯文本。LangChain 提供了 `with_structured_output` 方法，结合 **Pydantic** 实现强类型验证。

### 场景：提取电影信息

```python
from pydantic import BaseModel, Field

# 1. 定义数据 Schema (Pydantic)
class MovieDetails(BaseModel):
    """电影详细信息结构"""
    title: str = Field(description="电影标题")
    year: int = Field(description="发行年份")
    director: str = Field(description="导演姓名")
    tags: list[str] = Field(description="电影标签/流派")

# 2. 配置模型输出结构
structured_llm = model.with_structured_output(MovieDetails)

# 3. 调用模型
result = structured_llm.invoke("请介绍一下诺兰的《盗梦空间》")

# 4. 直接获取对象
print(type(result)) # <class '__main__.MovieDetails'>
print(result.title) # Inception (或中文名)
print(result.year)  # 2010
print(result.dict())
```

> **提示**: 相比传统的 Prompt Engineering (提示词工程) 要求返回 JSON，这种方式更稳定、更准确，且支持类型检查。

---

## 5. 高级特性 (Advanced Topics)

*   **多模态 (Multimodal)**: 支持传入图片、音频等多媒体内容（需模型支持，如 GPT-4o, Claude 3.5）。
*   **推理 (Reasoning)**: 部分模型（如 OpenAI o1）支持思维链推理，可获取 `reasoning_content`。
*   **运行时配置 (Configurable)**: 使用 `RunnableConfig` 在运行时动态修改参数。

```python
# 运行时动态切换模型配置示例
response = model.invoke(
    "Hello",
    config={
        "configurable": {"model": "gpt-4o-mini"}, # 动态覆盖模型
        "tags": ["test-run"], # 添加标签用于追踪
        "max_concurrency": 5  # 限制并发数
    }
)
```

---
*Last Updated: 2025*
