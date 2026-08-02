import json
import logging
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

import ast
import yaml
#python的日志

#把当前目录的.env加载进环境变量
load_dotenv(override= True)

#设置日志的基本信息
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
#在该程序中启动日志
logger = logging.getLogger(__name__)



#开启llm客户端

WORKDIR = Path.cwd()
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

SKILLS_DIR = WORKDIR/"skills"

#模型信息
MODEL = os.getenv("MODEL_ID", "deepseek-chat")
#给llm的初始提示词
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
)
#对于sub agent的初始提示词
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)

#todo_write, 模型的规划，让模型知道下一步该做什么，没做什么，目标是什么
CURRENT_TODOS = []

#组成提示词的部分。提取skill的meta，用sop来帮llm规划任务完成路径
def _parse_frontmatter(text: str):
    #不存在meta部分
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    #格式不正确
    if len(parts) < 3:
        return {}, text

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}

    return meta, parts[2].strip()

SKILL_REGISTRY = {}

#扫描所有的.md文件，然后组成skill注册表
def _scan_skills():
    #文件目录不存在
    if not SKILLS_DIR.exists():
        return
    #遍历每个skill
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        #找到skill.md的具体文件路径
        manifest = skill_dir / "SKILL.md"
        if not manifest.exists():
            continue

        raw = manifest.read_text()
        #从扫描文件函数找到meta， body
        meta, body = _parse_frontmatter(raw)

        #找到name
        name = meta.get("name", skill_dir.name)
        #找到description，没有就拿正文第一段
        description = meta.get(
            "description",
            body.split("\n")[0].lstrip("#").strip(),
        )
        #构建skills注册表
        SKILL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "content": raw,
        }

def list_skills() -> str:
    if not SKILL_REGISTRY:
        return "(no skills found)"

    lines = []
    for skill in SKILL_REGISTRY.values():
        lines.append(f"- {skill['name']}: {skill['description']}")

    return "\n".join(lines)

#用skills来重新组成提示词
def build_system() -> str:
    #每次都清空
    SKILL_REGISTRY.clear()
    #写入新的数据到注册表
    _scan_skills()

    skills_catalog = list_skills()

    return (
        f"You are a coding agent at {WORKDIR}. "
        "Before starting any multi-step task, use todo_write to plan your steps. "
        "Update status as you go.\n"
        f"Skills available:\n{skills_catalog}\n"
        "When a skill seems relevant, use load_skill to read its full instructions."
    )

#验证模型是否返回正确的todo
def _normalize_todos(todos):
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None


#写tools工具列表，给llm看
#第一个type代表了llm返回的参数必须是对象的形式，"xxx": "yyy"， 第二个单纯是代表了command时什么数据类型
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "Create and manage a task list for the current coding session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
]
#加载skill详细信息的函数
def load_skill(name: str) -> str:
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]


TOOLS.append({
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": "Load the full instructions for a skill by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
    },
})


#可以让llm返回一个todos来规划
def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS

    todos, error = _normalize_todos(todos)
    if error:
        return error

    CURRENT_TODOS = todos

    lines = ["\n## Current Tasks"]
    for todo in CURRENT_TODOS:
        icon = {
            "pending": " ",
            "in_progress": ">",
            "completed": "x",
        }[todo["status"]]
        lines.append(f"  [{icon}] {todo['content']}")

    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"

#确保路径安全，属于硬拒绝，不能写到外部
def safe_path(p: str) -> Path:
    #拼出路径
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

#阅读文件
def run_read(path: str, limit: Optional[int] = None) -> str:
    try:
        #用safe path保证路径正确
        lines = safe_path(path).read_text().splitlines()
        #限制取前limit行
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

#写文件
def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        #没有父文件的话一起创建父文件，文件已存在不报错
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

#替换文件内容
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        #拿到原文件内容
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        #替换
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

#找文件
def run_glob(pattern: str) -> str:
    try:
        results = []
        for match in WORKDIR.glob(pattern):
            resolved = match.resolve()
            if resolved.is_relative_to(WORKDIR):
                results.append(str(resolved.relative_to(WORKDIR)))
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

def run_bash(command: str):
    #异常拦截
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    #开始正式运行command,.run表示运行系统命令
    try:
        r = subprocess.run(
            command,
            shell= True,#用shell执行命令
            cwd= os.getcwd(),#在当下文件执行命令
            capture_output=True,#抓取输出
            text= True,#用字符串输出
            timeout= 120
        )
        #输出和非抓取的报错结合
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

#str name对应到具体的执行函数
TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_write": run_todo_write,
    "load_skill": load_skill,
}

#sub agent的可用工具，不能再给task，因为可能会无限递归
SUB_TOOLS = TOOLS[:5]

#sub agent的函数对应表
SUB_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


# bash 硬拒绝列表：命中这些片段就绝不执行。
DENY_LIST = [
    "rm -rf /",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "> /dev/sda",
]
#bash权限检测，属于硬拒绝
def check_deny_list(command: str) -> Optional[str]:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


#软拒绝规则，有风险要询问用户
PERMISSION_RULES = [
    {
        "tools": ["write_file", "edit_file"],
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Writing outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: any(
            kw in args.get("command", "")
            for kw in ["rm ", "> /etc/", "chmod 777"]
        ),
        "message": "Potentially destructive command",
    },
]

#软拒绝函数，检查是否存在风险行为
def check_rules(tool_name: str, args: dict) -> Optional[str]:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None

#软拒绝后的询问函数
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"

#综合的检查权限函数，要通过硬拒绝或者软拒绝才能返回true，再正常运行
def check_permission(tool_name: str, args: dict) -> bool:
    if tool_name == "bash":
        reason = check_deny_list(args.get("command", ""))
        if reason:
            print(f"\n\033[31m⛔ {reason}\033[0m")
            return False

    reason = check_rules(tool_name, args)
    if reason:
        decision = ask_user(tool_name, args, reason)
        if decision == "deny":
            return False

    return True

#提取message dict里面的content，返回一个str，拿出文本
def extract_text(message) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
        return "" if content is None else str(content)
    return "" if message is None else str(message)

#sub agent函数调用
def spawn_subagent(description: str)->str:
    print("\n[Subagent spawned]")
    #定义初始记忆
    messages = [{"role": "user", "content": description}]

    #固定循环次数，防止无限循环
    for _ in range(30):
        response = client.chat.completions.create(
            model= MODEL,
            messages= [{"role": "system", "content": SUB_SYSTEM}, *messages],
            tools= SUB_TOOLS,
            max_tokens= 8000,
        )

        message = response.choices[0].message #还是sdk，要转换
        assistant_message = message.model_dump(exclude_none= True)
        messages.append(assistant_message)

        if not message.tool_calls:
            break

        else:
            for tool_call in message.tool_calls:
                #拿到tool的基本信息
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")

                #跟主agent一样使用hook来完成检查
                blocked = trigger_hooks("PreToolUse", tool_name, tool_args)
                if blocked is not None:
                    messages.append(make_tool_result_message(tool_call_id= tool_call.id, output= str(blocked)))
                    continue

                handler = SUB_HANDLERS.get(tool_name)#拿到函数

                if handler is None:
                    output = f"Error: Unknown tool {tool_name}"
                else:
                    try:
                        output = handler(**tool_args)
                    
                    except Exception as e:
                        output = f"Error: {e}"

                trigger_hooks("PostToolUse", tool_name, tool_args, output)
                messages.append(make_tool_result_message(tool_call_id= tool_call.id, output= str(output)))

    #循环完毕，取最后一次message的content作为return
    print("[Subagent done]")
    return extract_text(messages[-1]) or "Subagent finished without a text response."


TOOLS.append({
    "type": "function",
    "function": {
        "name": "task",
        "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
            },
            "required": ["description"],
        },
    },
})
TOOL_HANDLERS["task"] = spawn_subagent

#压缩上下文，确定messages从哪里开始切开
COMPACT_TRIGGER_MESSAGES = 20 #messages最大长度
COMPACT_KEEP_TAIL = 5 #保留最近几条

#确定切开的尾部int
def _safe_tail_start(messages: list, keep_tail: int) -> int:
    tail_start = max(0, len(messages) - keep_tail)

    while tail_start > 0 and messages[tail_start].get("role") == "tool":
        tail_start -= 1

    return tail_start

#把要压缩的messages交给llm来压缩和返回
def compact_history(messages: list, keep_tail: int = COMPACT_KEEP_TAIL) -> list:
    if len(messages) <= COMPACT_TRIGGER_MESSAGES:
        return messages

    tail_start = _safe_tail_start(messages, keep_tail)
    old_messages = messages[:tail_start]
    recent_messages = messages[tail_start:]

    conversation = json.dumps(
        old_messages,
        ensure_ascii=False,
        default=str,
    )

    prompt = (
        "Summarize this coding-agent conversation so the work can continue.\n"
        "Preserve these details:\n"
        "1. current user goal\n"
        "2. important decisions and constraints\n"
        "3. files read or changed\n"
        "4. tool results that still matter\n"
        "5. remaining work\n\n"
        "Conversation:\n"
        f"{conversation[:80000]}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You summarize conversation history. Do not call tools.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=2000,
    )

    summary = response.choices[0].message.content or "(empty summary)"
    summary_message = {
        "role": "user",
        "content": f"[Compacted Summary]\n\n{summary}",
    }

    return [summary_message, *recent_messages]


def make_tool_result_message(tool_call_id: str, output: str):
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": output
    }

#agent loop过于庞大，加入hook来缩减流程
HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": []
}
#注册hook函数
def register_hook(event: str, callback):
    HOOKS[event].append(callback)

#运行hook里的阶段函数
def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        #有问题才会返回非none
        if result is not None:
            return result
    return None

#定义_hook函数，把原agent loop里的流程包成一个函数，放进hooks里面
def permission_hook(tool_name: str, args: dict):
    if not check_permission(tool_name, args):
        return "Permission denied."
    return None


def log_hook(tool_name: str, args: dict):
    args_preview = str(list(args.values())[:2])[:60]
    print(f"\033[90m[HOOK] {tool_name}({args_preview})\033[0m")
    return None


def large_output_hook(tool_name: str, args: dict, output: str):
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] Large output from {tool_name}: {len(str(output))} chars\033[0m")
    return None

def context_inject_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None

def summary_hook(messages: list):
    tool_count = sum(
        1
        for m in messages
        if m.get("role") == "tool"
    )
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


#开始agent loop
rounds_since_todo = 0
def agent_loop(messages: list):
    global rounds_since_todo
    while True:
        #压缩上下文
        messages[:] = compact_history(messages)
        #加入计划表更新计数器
        if rounds_since_todo >= 3 and messages:
            messages.append({
                "role": "user",
                "content": "<reminder>Update your todos.</reminder>"
            })
            rounds_since_todo = 0
        #输出日志
        logger.info("requesting model=%s messages=%d", MODEL, len(messages))
        #连接llm获得回答
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": build_system()}, *messages],
            tools=TOOLS,
            max_tokens=8000,
        )
        
        # SDK 返回的是对象；这里把 assistant message 转成 dict，方便加入 messages。
        message = response.choices[0].message
        assistant_message = message.model_dump(exclude_none=True)

        # 追加 assistant 原始消息，里面可能带有 tool_calls。
        messages.append(assistant_message)

        # DeepSeek/OpenAI 用 tool_calls 表示模型要继续调用工具。
        #如果回答里没有tool call，直接返回
        if not message.tool_calls:
            logger.info("agent loop finished: finish_reason=%s", response.choices[0].finish_reason)
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": str(force)})
                continue

            return

        rounds_since_todo += 1 #计数器，一次查询+1
        #有tool call就要调用所有，并把每次的name和output加进短期记忆里以便llm再次判断
        for tool_call in message.tool_calls:
            #拿到对应函数的参数
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            #用hook来走pretooluse流程
            blocked = trigger_hooks("PreToolUse", tool_name, tool_args)
            if blocked:
                messages.append(make_tool_result_message(tool_call_id= tool_call.id, output= str(blocked)))
                continue

            #得到具体函数
            handler = TOOL_HANDLERS.get(tool_name)
            
            if handler is None:
                logger.warning("unknown tool requested: %s", tool_call.function.name)
                output = f"Error: Unknown tool {tool_call.function.name}"
                
            else:
                try:
                    output = handler(**tool_args)
                    if tool_name == "todo_write":
                        rounds_since_todo = 0
                except Exception as e:
                    output = f"Error: {e}"
            
            trigger_hooks("PostToolUse", tool_name, tool_args, output)
            print(str(output)[:200])
            # 工具结果必须用 role=tool 和 tool_call_id 回填，模型才能接着推理。
            messages.append(make_tool_result_message(tool_call.id, str(output)))

if __name__ == "__main__":
    print("Harness Agent")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    
    while True:
        try:
            query = input("\033[36magent >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the model's final text response
        response_content = history[-1].get("content")
        if response_content:
            print(response_content)
        print()
