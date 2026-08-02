# Harness Agent

一个轻量级命令行 agent harness，基于 OpenAI-compatible Chat Completions 接口实现。

它包含：

- tool calling
- 文件读写和 shell 执行工具
- 工具权限检查
- hook 扩展点
- todo 规划工具
- subagent 调度
- skill 按需加载
- 简单上下文压缩

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填入 `DEEPSEEK_API_KEY`。

## Usage

```sh
python agent.py
```

输入问题后，agent 会根据模型返回的 tool calls 执行工具，并把工具结果回填到上下文中继续推理。

## Skills

skill 放在 `skills/<skill-name>/SKILL.md`，文件头部可以包含 YAML frontmatter：

```md
---
name: code-review
description: Review code changes for bugs and risks.
---
```

启动请求前，agent 会把 skill 摘要注入 system prompt；当模型需要完整流程时，可以调用 `load_skill` 读取全文。
