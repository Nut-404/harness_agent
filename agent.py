import json
import logging
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

import time
import re
import random

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


#获取工作目录，项目父目录
WORKDIR = Path.cwd()
#开启llm客户端
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

#记忆文件路径，在父目录下
MEMORY_DIR = WORKDIR / ".memory"
#具体记忆文件，总表
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
#确保存在记忆文件路径
MEMORY_DIR.mkdir(exist_ok=True)
#设置记忆类型
MEMORY_TYPES = ("user", "feedback", "project", "reference")

#skill文件路径
SKILLS_DIR = WORKDIR/"skills"

#模型信息
MODEL = os.getenv("MODEL_ID", "deepseek-chat")
#设置最大token数量
DEFAULT_MAX_TOKENS = 8000
#默认token不能满足
ESCALATED_MAX_TOKENS = 16000
#设置最大token尝试次数
MAX_CONTINUATION_RETRIES = 3
#给token错误后的提示词
CONTINUATION_PROMPT = (
    "Output token limit hit. Continue directly from where you stopped. "
    "Do not repeat earlier content."
)
#重新尝试的次数
MAX_RETRIES = 3
#重新尝试的延迟量
BASE_DELAY_MS = 500
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
#task, 目标是什么，todo是任务，是步骤，task是目标
TASKS = {}
CURRENT_TASK_ID = None
TASK_STATUSES = ("pending", "in_progress", "completed", "failed")

#task id 生成
def make_task_id()->str:
    return f"task-{int(time.time())}"

#用任务描述创建任务
def create_task(description: str):
    global CURRENT_TASK_ID

    task_id = make_task_id()
    now = int(time.time())

    #创建任务
    task = {
        "id": task_id,
        "description": description,
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "result": ""
    }

    CURRENT_TASK_ID = task_id
    TASKS[task_id] = task
    return task

#更新task
def update_task(task_id: str, status: str, result: str = "") -> str:
    if task_id not in TASKS:
        return f"Error: task not found: {task_id}"

    if status not in TASK_STATUSES:
        return f"Error: invalid task status: {status}"

    task = TASKS[task_id]
    task["status"] = status
    task["updated_at"] = int(time.time())

    if result:
        task["result"] = result

    return f"Updated task {task_id} to {status}"

#读取当前任务，返回任务
def get_current_task():
    if CURRENT_TASK_ID is None:
        return None

    return TASKS.get(CURRENT_TASK_ID)

#组成提示词的部分。提取skill和memory的meta，用sop来帮llm规划任务完成路径
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

#记忆模块，从llm提取出的名字里得到新的memory文字来作为memory文件的文件名
def _memory_slug(name: str) -> str:
    slug = name.lower().strip()
    slug = slug.replace(" ", "-").replace("/", "-")
    return slug or "memory"

#写memory文件
def write_memory_file(name: str, mem_type: str, description: str, body: str) -> str:
    #非规定类型，设定默认值
    if mem_type not in MEMORY_TYPES:
        mem_type = "user"

    slug = _memory_slug(name)
    filename = f"{slug}.md"
    filepath = MEMORY_DIR / filename
    #把数据写进去
    filepath.write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"type: {mem_type}\n"
        f"---\n\n"
        f"{body}\n"
    )

    #改动记忆文件，重新刷新给llm的记忆表
    _rebuild_index()
    return f"Wrote memory: {filename}"

#建立记忆表，跟skill一样，包含name和description
def _rebuild_index():
    lines = []

    for memory_file in sorted(MEMORY_DIR.glob("*.md")):
        if memory_file.name == "MEMORY.md":
            continue

        #读取text并拆分
        raw = memory_file.read_text()
        meta, body = _parse_frontmatter(raw)

        #这里meta已经是dict
        name = meta.get("name", memory_file.stem)
        description = meta.get("description", body.split("\n")[0][:80])

        lines.append(f"- [{name}]({memory_file.name}) - {description}")

    content = "\n".join(lines)
    if content:
        content += "\n"
    #memory.md为总表
    MEMORY_INDEX.write_text(content)

#阅读这个memory的总表，里面有memories的name和description
def read_memory_index() -> str:
    if not MEMORY_INDEX.exists():
        return ""

    return MEMORY_INDEX.read_text().strip()

#阅读某一个记忆文件
def read_memory_file(filename: str) -> str:
    memory_path = (MEMORY_DIR / filename).resolve()

    if not memory_path.is_relative_to(MEMORY_DIR.resolve()):
        return f"Error: memory path escapes memory directory: {filename}"

    if not memory_path.exists():
        return f"Error: memory file not found: {filename}"

    return memory_path.read_text()

#列出所有的记忆文件的详细信息
def list_memory_files() -> list:
    memories = []

    for memory_file in sorted(MEMORY_DIR.glob("*.md")):
        if memory_file.name == "MEMORY.md":
            continue

        raw = memory_file.read_text()
        meta, body = _parse_frontmatter(raw)

        memories.append({
            "filename": memory_file.name,
            "name": meta.get("name", memory_file.stem),
            "description": meta.get("description", ""),
            "type": meta.get("type", "user"),
            "body": body,
        })

    return memories


#选出具体memory信息。参数messages时短期的对话记忆。参考用户之前三次说的来推出当前需要什么memory文件
def select_relevant_memories(messages: list, max_items: int = 5) -> list:
    #得到详细的记忆文件列表
    files = list_memory_files()
    if not files:
        return []
    #提取最近三次的user输入
    recent_parts = []
    for msg in reversed(messages):
        #如果是用户说的
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                recent_parts.append(content)

        if len(recent_parts) >= 3:
            break

    #用户输入组合成一个str
    recent_text = " ".join(reversed(recent_parts)).lower()
    if not recent_text.strip():
        return []

    #这个keywords怎么得出来的，为什么这样算
    keywords = []
    for word in recent_text.replace("-", " ").replace("_", " ").split():
        word = word.strip(".,!?;:()[]{}'\"").lower()
        if len(word) >= 4:
            keywords.append(word)

    selected = []
    for memory in files:
        searchable = (
            memory["name"] + " " +
            memory["description"] + " " +
            memory["body"]
        ).lower()

        if any(keyword in searchable for keyword in keywords):
            selected.append(memory["filename"])

        if len(selected) >= max_items:
            break

    return selected

#agent loop完成后，对messages[-10:]做记忆提取和更新
def extract_memories(messages: list)->None:
    #最晚的十条作为记忆提取目标
    dialogue_parts = []
    for msg in messages[-10:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        #确保content不为空
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content}")
    #变成str
    dialogue = "\n".join(dialogue_parts)
    if not dialogue.strip():
        return
    #得到所有的memory详细信息
    existing = list_memory_files()
    #变成str
    existing_text = "\n".join(
        f"- {m['name']}: {m['description']}"
        for m in existing
    ) or "(none)"

    #构建pormpt
    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n"
        "- name: short kebab-case identifier, e.g. 'user-preference-tabs'\n"
        "- type: one of 'user', 'feedback', 'project', 'reference'\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown\n"
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_text}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )

    #开始问llm
    try:

        response = client.chat.completions.create(
            model= MODEL,
            messages= [
                {
                    "role": "system",
                    "content": "You extract durable agent memory as JSON. Do not call tools.",
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            max_tokens= 800
        )

        #拿到答案
        text = response.choices[0].message.content or "[]"
        #正则查找，查找左边为[, 右边为]，中间任意的结构。我们希望返回list
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return

        #把返回的json变成python list
        items = json.loads(match.group())
        if not isinstance(items, list):
            return

        count = 0
        for item in items:
            #格式错误
            if not isinstance(item, dict):
                continue

            #items是list， item是dict
            name = item.get("name") or f"memory-{int(time.time())}"#拿到名字，没有就用时间戳代替
            mem_type = item.get("type", "user")
            description = item.get("description", "")
            body = item.get("body", "")

            #这两个必须存在
            if description and body:
                write_memory_file(name, mem_type, description, body)
                count += 1

        if count:
            print(f"[Memory: extracted {count} new memories]")

    except Exception as e:
        logger.warning("memory extraction failed: %s", e)


#记忆文件太多后，需要去重，降低数量
#memory记忆数量上限
CONSOLIDATE_THRESHOLD = 10
#依旧使用llm来总结memory并写入
def consolidate_memories() -> None:
    #拿到所有的记忆文件细节
    files = list_memory_files()

    if len(files) < CONSOLIDATE_THRESHOLD:
        return

    #拼接所有的细节
    catalog = "\n\n".join(
        f"## {f['filename']}\n"
        f"name: {f['name']}\n"
        f"type: {f['type']}\n"
        f"description: {f['description']}\n\n"
        f"{f['body']}"
        for f in files
    )

    prompt = (
        "Consolidate these memory files.\n"
        "Rules:\n"
        "1. Merge duplicate memories.\n"
        "2. Remove outdated or contradicted memories.\n"
        "3. Preserve important user preferences and project constraints.\n"
        "4. Keep useful details in body.\n"
        "Return ONLY a JSON array. Each item must be:\n"
        "{'name': str, 'type': str, 'description': str, 'body': str}\n\n"
        f"{catalog[:16000]}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You consolidate durable agent memories as JSON. Do not call tools.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=3000,
        )

        text = response.choices[0].message.content or "[]"
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return

        items = json.loads(match.group())
        if not isinstance(items, list):
            return

        for memory_file in MEMORY_DIR.glob("*.md"):
            if memory_file.name != "MEMORY.md":
                memory_file.unlink()

        count = 0
        for item in items:
            if not isinstance(item, dict):
                continue

            name = item.get("name") or f"memory-{int(time.time())}"
            mem_type = item.get("type", "user")
            description = item.get("description", "")
            body = item.get("body", "")

            if description and body:
                write_memory_file(name, mem_type, description, body)
                count += 1

        print(f"[Memory: consolidated {len(files)} memories into {count}]")

    except Exception as e:
        logger.warning("memory consolidation failed: %s", e)


#skill注册表
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

#用skill注册表创建概括性的总结str
def list_skills() -> str:
    if not SKILL_REGISTRY:
        return "(no skills found)"

    lines = []
    for skill in SKILL_REGISTRY.values():
        lines.append(f"- {skill['name']}: {skill['description']}")

    return "\n".join(lines)

#用skills和memory来重新组成提示词
#用来检查是否还需要拼接，或者和上次数据一样直接返回
LAST_SYSTEM_KEY = None
LAST_SYSTEM_PROMPT = None
#创建初始提示词
def build_system(memories: str = "") -> str:
    global LAST_SYSTEM_KEY, LAST_SYSTEM_PROMPT
    # 每次请求前重新扫描 skill 目录，让新增/修改的 skill 生效。
    SKILL_REGISTRY.clear()
    _scan_skills()
    #列出所有的skill
    skills_catalog = list_skills()
    #memory.md文件总结
    memory_index = read_memory_index()
    #列出所有支持的工具能力
    enabled_tools = ",".join(TOOL_HANDLERS.keys())
    #给出任务
    current_task = get_current_task()

    #组装system_key
    system_key = json.dumps(
        {
            "workdir": str(WORKDIR),
            "skills_catalog": skills_catalog,
            "memory_index": memory_index,
            "memories": memories,
            "enable_tools": enabled_tools,
            "current_task": current_task
        },
        sort_keys= True,
        ensure_ascii= False
    )
    #区分一下是否数据改变，是否需要重新拼接
    if system_key == LAST_SYSTEM_KEY and LAST_SYSTEM_PROMPT:
        return LAST_SYSTEM_PROMPT
    #分区块来组装prompt
    sections = []
    #要求工作地区，工作目录，限制工作范围
    sections.append(
        f"You are a coding agent at {WORKDIR}. "
        "Use tools to solve coding tasks carefully."
    )
    #要求任务步骤，要先写todo
    sections.append(
        "Before starting any multi-step task, use todo_write to plan your steps. "
        "Update status as you go."
    )
    #展现可用工具
    sections.append(
        "Available tool names:\n"
        f"{enabled_tools}"
    )
    #展示当前任务
    if current_task:
        sections.append(
            "Current task: \n"
            f"- id: {current_task['id']}\n"
            f"- status: {current_task['status']}\n"
            f"- description: {current_task['description']}"
        )
    #展现可用技能
    sections.append(
        "Skills available:\n"
        f"{skills_catalog}\n"
        "When a skill seems relevant, use load_skill to read its full instructions."
    )

    if memory_index:
        sections.append(
            "Memories available:\n"
            f"{memory_index}"
        )

    if memories:
        sections.append(
            "Relevant memory details:\n"
            f"{memories}"
        )

    sections.append(
        "Memory rules:\n"
        "- Memories are long-term user preferences, feedback, project facts, or references.\n"
        "- Use relevant memory details when they apply to the current task.\n"
        "- Do not treat todo items or compact summaries as long-term memory."
    )

    #替代key，下次检查
    LAST_SYSTEM_KEY = system_key
    #替代
    LAST_SYSTEM_PROMPT = "\n\n".join(sections)
    return LAST_SYSTEM_PROMPT

#验证模型是否返回正确的todo
#llm应该会返回怎么样的todo list？为什么会返回，是我们在提示词里要求的吗？
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
    {
        "type": "function",
        "function": {
            "name": "task_status",
            "description": "Show the current task state.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function":{
            "name": "task_update",
            "description": "Update the current task status and optional result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "failed"],
                    },
                    "result": {"type": "string"}
                },
                "required": ["status"]
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
#这个todo部分的数据流是怎么样的？
def run_todo_write(todos: list) -> str:
    #全局todo list
    global CURRENT_TODOS
    #确实todo list确实是list格式
    todos, error = _normalize_todos(todos)
    if error:
        return error

    CURRENT_TODOS = todos

    lines = ["\n## Current Tasks"]
    for todo in CURRENT_TODOS:
        #根据status来写icon
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

#llm获取task的函数
def run_task_status()->str:
    task = get_current_task()
    if not task:
        return "No current task"
    return json.dumps(task, ensure_ascii= False, indent= 2)

#llm更新task
def run_task_update(status: str, result: str = "")->str:
    task = get_current_task()
    if not task:
        return "Error: no current task"
    return update_task(task["id"], status, result)

#str name对应到具体的执行函数
TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_write": run_todo_write,
    "load_skill": load_skill,
    "task_status": run_task_status,
    "task_update": run_task_update,
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
    #每条软拒绝规则检查
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

#综合的检查权限函数，要通过硬拒绝和软拒绝才能返回true，再正常运行
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
#为sub agent服务，用于作答时用最后一次的回答作为答案
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
                #sub agent同样需要检验
                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    output = f"Error: Invalid JSON arguments for {tool_name}: {e}"
                    messages.append(make_tool_result_message(tool_call_id= tool_call.id, output= output))
                    continue
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

    #确定尾部切断位置
    tail_start = _safe_tail_start(messages, keep_tail)
    #切成两半
    old_messages = messages[:tail_start]
    recent_messages = messages[tail_start:]

    #让old_message变成str格式
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

#压缩history后，还是可能会导致上下文过大，用来判断是否是上下文过大导致llm返回错误
def is_prompt_too_long_error(e: Exception):
    msg = str(e).lower()
    return (
        "prompt_too_long" in msg
        or "prompt is too long" in msg
        or "context_length_exceeded" in msg
        or "maximum context length" in msg
        or "max_context_window" in msg
    )

#粗略压缩，实验是否是压缩问题，简略压缩后如还有上下文问题，就不是压缩的问题
def reactive_compact(messages: list):
    #尾部卡在对话数组第几位
    tail_start = _safe_tail_start(messages= messages, keep_tail= 5)
    #留存对话
    recent_messages = messages[tail_start:]
    #直接丢弃不总结
    return [
        {
            "role": "user",
            "content": "[Reactive compact] Earlier conversation was trimmed because the prompt was too long. Continue from the remaining recent context.",
        },
        *recent_messages,
    ]

#对于请求暂时不可用进行处理
#创建随机时间
def retry_delay(attempt: int)->float:
    #基础延时
    base = min(BASE_DELAY_MS * (2**attempt), 32000)/1000
    #加入随机量
    jitter = random.uniform(0, base*0.25)

    return base+jitter

#识别是否是模型暂时不能响应
def is_transient_model_error(e: Exception):
    #错误信息
    msg = str(e).lower()
    #找到错误名称
    name = type(e).__name__.lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "ratelimit" in name
        or "529" in msg
        or "overloaded" in msg
        or "timeout" in msg
        or "temporarily unavailable" in msg
    )

#重新尝试函数，fn为函数名
def with_retry(fn):
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            #不是暂时的连接问题
            if not is_transient_model_error(e):
                raise
            #添加等待量
            delay = retry_delay(attempt= attempt)
            logger.warning(
                "transient model error; retrying %s/%s after %.1fs: %s",
                attempt + 1,
                MAX_RETRIES,
                delay,
                e,
            )
            time.sleep(delay)
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded")




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
    #只能失误一次，压缩过一次之后还不能通过说明是单条memory或者消息太大，再粗糙压缩没有意义
    attempted_reactive_compact = False
    #单次对话的token数量更改bool
    has_escalated_max_tokens = False
    max_tokens = DEFAULT_MAX_TOKENS
    continuation_retries = 0
    while True:
        
        #压缩上下文
        messages[:] = compact_history(messages)
        #要搜寻相关记忆在记忆文件内
        selected_memories = select_relevant_memories(messages)
        memories = ""
        memory_parts = []
        for memory_name in selected_memories:
            memory_parts.append(read_memory_file(memory_name))

        memories = "\n\n".join(memory_parts)
        
        #加入计划表更新计数器
        if rounds_since_todo >= 3 and messages:
            messages.append({
                "role": "user",
                "content": "<reminder>Update your todos.</reminder>"
            })
            rounds_since_todo = 0
        #输出日志
        logger.info("requesting model=%s messages=%d", MODEL, len(messages))
        try:
            #连接llm获得回答
            response = with_retry(
                lambda: client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": build_system(memories=memories)}, *messages],
                    tools=TOOLS,
                    max_tokens=max_tokens,
                )
            )
        except Exception as e:
            #如果是上下文太长的问题并且还没有粗略压缩过
            if is_prompt_too_long_error(e) and not attempted_reactive_compact:
                #日志报错
                logger.warning("prompt too long; running reactive compact and retrying")
                #之前没粗略压缩过，现在压缩
                messages[:] = reactive_compact(messages)
                #更改压缩记录
                attempted_reactive_compact = True
                continue
            #压缩过了，直接报错
            logger.warning("model request failed: %s", e)
            #直接加入短期记忆
            messages.append({
                "role": "assistant",
                "content": f"Error: model request failed: {e}",
            })
            #更新任务状态为failed
            current_task = get_current_task()
            if current_task and current_task["status"] == "in_progress":
                update_task(
                    current_task["id"],
                    "failed",
                    f"model request failed: {e}",
                )
            return
        
        # SDK 返回的是对象；这里把 assistant message 转成 dict，方便加入 messages。
        message = response.choices[0].message
        assistant_message = message.model_dump(exclude_none=True)

        # 追加 assistant 原始消息，里面可能带有 tool_calls。
        messages.append(assistant_message)

        #看是否是因为token限制导致回答不完整
        finish_reason = response.choices[0].finish_reason

        #如果是token原因导致停止
        if finish_reason == "length":
            #如果还没有更改过最大限制token数
            if not has_escalated_max_tokens:
                #尝试更改token数能不能获得完整答案
                max_tokens = ESCALATED_MAX_TOKENS
                has_escalated_max_tokens = True
                #垃圾信息要删除
                messages.pop()
                logger.warning(
                    "max_tokens hit; escalating %s -> %s",
                    DEFAULT_MAX_TOKENS,
                    ESCALATED_MAX_TOKENS
                )
                #直接进入下一轮循环，重新获得输出
                continue
            #在合理的尝试次数内还是要加入messages
            if continuation_retries < MAX_CONTINUATION_RETRIES:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                continuation_retries+=1
                logger.warning(
                    "max_tokens hit after escalation; requesting continuation %s/%s",
                    continuation_retries,
                    MAX_CONTINUATION_RETRIES
                )
                continue

            logger.warning("max_tokens recovery limit reached")
            return

        # DeepSeek/OpenAI 用 tool_calls 表示模型要继续调用工具。
        #如果回答里没有tool call，直接返回
        if not message.tool_calls:
            logger.info("agent loop finished: finish_reason=%s", response.choices[0].finish_reason)
            extract_memories(messages= messages)
            consolidate_memories()
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": str(force)})
                continue
            #更新任务状态
            current_task = get_current_task()
            if current_task and current_task["status"] == "in_progress":
                update_task(
                    current_task["id"],
                    "completed",
                    extract_text(assistant_message)
                )

            return

        rounds_since_todo += 1 #计数器，一次查询+1
        #有tool call就要调用所有，并把每次的name和output加进短期记忆里以便llm再次判断
        for tool_call in message.tool_calls:
            #拿到对应函数的参数
            tool_name = tool_call.function.name
            #args可能没有返回正常结果, 如果错误结果要把错误消息传回去等待再次调用
            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                output = f"Error: invalid JSON arguments for {tool_name}: {e}"
                messages.append(make_tool_result_message(tool_call_id= tool_call.id, output= output))
                continue

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
        #加入任务
        if get_current_task() is None:
            task = create_task(query)
            print(f"[Task created] {task['id']}: {task['description']}")
        agent_loop(history)
        #输出最后一次作答作为答案
        response_content = history[-1].get("content")
        if response_content:
            print(response_content)
        print()
