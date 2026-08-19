# Harness Agent

一个教学版命令行 coding agent harness，基于 OpenAI-compatible Chat Completions 接口实现。项目重点不是做一个完整产品，而是把 agent 外层的控制系统一点点手搓出来：工具、权限、上下文、任务、团队协作、worktree 隔离和 MCP 外部工具接入。

## 功能

- Agent loop：模型返回普通回答时结束；返回 tool calls 时执行工具并回填结果继续推理。
- Tool use：支持 `bash`、文件读写、精确替换、glob 搜索等基础工具。
- Permission：硬拒绝危险 shell 命令，软规则会询问用户；MCP deploy 类工具也会进入审批。
- Hooks：提供 `UserPromptSubmit`、`PreToolUse`、`PostToolUse`、`Stop` 生命周期扩展点。
- Todo：`todo_write` 用于当前任务内的短期计划和状态更新。
- Task system：维护当前任务和共享任务板，支持创建、认领、完成任务。
- Background tasks：慢操作可以放到后台线程，完成后把结果注入后续对话。
- Cron scheduler：支持五段 cron 表达式，把定时 prompt 注入 agent loop。
- Subagent：`task` 工具可以启动一次性子 agent，隔离上下文后只返回最终总结。
- Skills：扫描 `skills/<name>/SKILL.md`，system prompt 注入技能目录，需要时用 `load_skill` 读取全文。
- Memory：把长期偏好、项目事实、反馈和引用保存到 `.memory/`，并在相关任务中注入。
- Context compact：自动压缩长对话，也提供 `compact` 工具让模型主动请求压缩。
- Agent teams：Lead 可以启动 teammate 线程，通过 `.mailboxes/` 文件邮箱通信。
- Team protocols：支持 request/response 形式的 shutdown、plan approval 等协议。
- Autonomous teammates：队友空闲时可以扫描任务板、自动认领可执行任务。
- Worktree isolation：任务可绑定独立 git worktree，队友在隔离目录里执行文件和 shell 工具。
- MCP tools：通过 mock MCP server 动态发现外部工具，工具名格式为 `mcp__server__tool`。

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS / Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填入：

```env
DEEPSEEK_API_KEY=sk-xxx
MODEL_ID=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## Usage

```sh
python agent.py
```

输入问题后，agent 会把用户消息加入 `history`，请求模型。如果模型返回 tool calls，harness 会执行工具、做权限检查、回填 tool result，然后继续下一轮模型请求。

可以尝试：

```text
列出这个项目里的 Python 文件，并总结 agent loop 的结构。
```

```text
Connect to the docs MCP server and search for agent loop.
```

```text
Create two tasks, create worktrees for them, then spawn alice and bob.
```

## MCP Mock Servers

当前内置两个 mock MCP server：

- `docs`：提供 `search` 和 `get_version`，模拟只读文档服务。
- `deploy`：提供 `trigger` 和 `status`，模拟部署服务；deploy 类 MCP 工具会触发权限询问。

连接后，MCP 工具会动态加入工具池：

```text
connect_mcp("docs")
-> mcp__docs__search
-> mcp__docs__get_version
```

## Skills

skill 放在 `skills/<skill-name>/SKILL.md`，文件头部可以包含 YAML frontmatter：

```md
---
name: code-review
description: Review code changes for bugs and risks.
---
```

启动请求前，agent 会把 skill 摘要注入 system prompt；当模型需要完整流程时，可以调用 `load_skill` 读取全文。

## Runtime Files

运行过程中可能生成这些本地目录：

- `.memory/`：长期记忆文件和索引。
- `.mailboxes/`：多 agent 团队通信邮箱。
- `.worktrees/`：git worktree 隔离目录和事件日志。
- `.task_outputs/`：大工具输出持久化目录。
- `.transcripts/`：长上下文转储目录。

这些目录是本地运行状态，不属于核心源码。

## Architecture

核心循环始终是：

```text
准备上下文
  -> 调用 LLM
  -> 如果没有 tool calls，Stop 收尾
  -> 如果有 tool calls，执行工具
  -> 回填 tool results
  -> 再次调用 LLM
```

各机制在循环中的位置：

- LLM 前：compact、cron、background、inbox、memory、system prompt、dynamic tool pool。
- Tool 前：JSON 参数解析、`compact` 特殊处理、permission、hooks。
- Tool 中：builtin handler、MCP handler、background dispatch。
- Tool 后：`PostToolUse` hooks、tool result 回填。
- Stop：memory extraction、memory consolidation、task completion、stop hooks。
