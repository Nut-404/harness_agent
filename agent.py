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
import threading
from datetime import datetime
from dataclasses import dataclass, field

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

#background后台进程记录
#记录详细的后台任务信息，bg_id -> {tool_name, args, status, started_at}
BACKGROUND_TASKS = {}
#bg_id -> 工具执行结果字符串
BACKGROUND_RESULTS = {}
#多线程同时读写 dict 时加锁
BACKGROUND_LOCK = threading.Lock()
#用来生成 bg_id
BACKGROUND_COUNTER = 0

#定时任务记录
#已经定好的任务
SCHEDULED_JOBS = {}
#已经到了时间但是还没运行的任务
CRON_QUEUE = []
#防止同一任务多次触发
LAST_FIRED = {}
#保护线程的锁
CRON_LOCK = threading.Lock()
#定时任务id
CRON_COUNTER = 0

#多agent协作
#通信邮箱地址
MAILBOX_DIR = WORKDIR / ".mailboxes"
MAILBOX_DIR.mkdir(exist_ok= True)
#可用或已经存在的teammate agent
ACTIVE_TEAMMATES = {}

#teammate agent扫描任务板的间隔
IDLE_POLL_INTERVAL = 5
#没找到任务时就sleep
IDLE_TIMEOUT = 60

#每个teammate要有自己的工作文件，worktree，避免相互覆盖
# worktree 隔离目录
WORKTREES_DIR = WORKDIR / ".worktrees"
WORKTREES_DIR.mkdir(exist_ok=True)

# 只允许简单安全的 worktree 名字，避免 ../ 这种路径穿越
VALID_WORKTREE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# 确认是合法路径
def validate_worktree_name(name: str) -> str | None:
    if not name:
        return "Worktree name cannot be empty"

    if name in (".", ".."):
        return f"Invalid worktree name: {name}"

    if not VALID_WORKTREE_NAME.match(name):
        return (
            f"Invalid worktree name '{name}': "
            "only letters, digits, dots, underscores, and dashes are allowed"
        )

    return None

#统一执行git worktree指令
def run_git(args: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        if not output:
            output = "(no output)"

        return result.returncode == 0, output[:5000]
    except subprocess.TimeoutExpired:
        return False, "Error: git timeout"

#记录worktree操作动作，方便追踪
def log_worktree_event(event_type: str, worktree_name: str, task_id: str = ""):
    event = {
        "type": event_type,
        "worktree": worktree_name,
        "task_id": task_id,
        "ts": time.time(),
    }

    events_file = WORKTREES_DIR / "events.jsonl"
    with events_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

#将任务绑定到worktree上
def bind_task_to_worktree(task_id: str, worktree_name: str):
    task = TASKS.get(task_id)
    if not task:
        return f"Error: task not found: {task_id}"

    task["worktree"] = worktree_name
    task["updated_at"] = int(time.time())

    return f"Bound task {task_id} to worktree {worktree_name}"

#创建worktree文件
def create_worktree(name: str, task_id: str = "") -> str:
    error = validate_worktree_name(name)
    if error:
        return f"Error: {error}"

    path = WORKTREES_DIR / name
    if path.exists():
        return f"Error: worktree already exists: {name}"

    ok, output = run_git([
        "worktree",
        "add",
        str(path),
        "-b",
        f"wt/{name}",
        "HEAD",
    ])

    if not ok:
        return f"Git error: {output}"

    if task_id:
        bind_result = bind_task_to_worktree(task_id, name)
        if bind_result.startswith("Error:"):
            return bind_result

    log_worktree_event("create", name, task_id)
    return f"Created worktree {name} at {path}"

# 对于多agent之间消息格式的约束
# 协议状态：记录 lead 和 teammate 之间正在等待回应的请求
@dataclass
class ProtocolState:
    request_id: str
    type: str  # "shutdown" | "plan_approval"
    sender: str
    target: str
    status: str  # pending | approved | rejected
    payload: str
    created_at: float = field(default_factory=time.time)


# request_id -> ProtocolState, 等待回应的请求
PENDING_REQUESTS = {}


# 创建协议请求 id，让请求和响应能对应起来
def make_request_id() -> str:
    return f"req_{random.randint(0, 999999):06d}"

# 收到协议响应后，用 request_id 找到原请求，并更新状态
def match_response(response_type: str, request_id: str, approve: bool) -> str:
    state = PENDING_REQUESTS.get(request_id)
    if not state:
        return f"Error: unknown request_id: {request_id}"

    # 防止 shutdown_response 错误地审批 plan_approval 请求
    if state.type == "shutdown" and response_type != "shutdown_response":
        return f"Error: type mismatch: expected shutdown_response, got {response_type}"

    if state.type == "plan_approval" and response_type != "plan_approval_response":
        return f"Error: type mismatch: expected plan_approval_response, got {response_type}"

    if state.status != "pending":
        return f"Request {request_id} already {state.status}"

    state.status = "approved" if approve else "rejected"
    return f"Protocol request {request_id} {state.status}"

#多agent的消息发送函数
def send_message(
    from_agent: str,
    to_agent: str,
    content: str,
    msg_type: str = "message",
    metadata: dict | None = None,
):
    #创建消息
    message = {
        "from": from_agent,
        "to": to_agent,
        "content": content,
        "type": msg_type,
        "metadata": metadata or {},
        "ts": time.time()
    }
    #to agent的邮箱地址
    inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
    #写入
    with inbox.open("a", encoding= "utf-8") as f:
        f.write(json.dumps(message, ensure_ascii= False) + "\n")

    return f"send message to {to_agent}"

#多agent的接收消息
def read_inbox(agent: str)->list:
    inbox = MAILBOX_DIR / f"{agent}.jsonl"
    if not inbox.exists():
        return []
    messages = []
    for line in inbox.read_text(encoding= "utf-8").splitlines():
        if not line.strip():
            continue

        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            messages.append(
                {
                    "from": "system",
                    "to": agent,
                    "content": f"Error: invalid inbox line : {line}",
                    "type": "error",
                    "ts": time.time()
                }
            )
    #阅读完就清空
    inbox.unlink()
    return messages


# 统一读取 lead inbox：先处理协议响应，再把消息交给 LLM 看。
def consume_lead_inbox(route_protocol: bool = True) -> list:
    messages = read_inbox("lead")
    if not messages:
        return []

    if route_protocol:
        for message in messages:
            metadata = message.get("metadata", {})
            request_id = metadata.get("request_id", "")
            msg_type = message.get("type", "")
            if request_id and msg_type.endswith("_response"):
                match_response(
                    response_type=msg_type,
                    request_id=request_id,
                    approve=metadata.get("approve", False),
                )

    return messages

#创建定时任务id
def make_cron_id():
    global CRON_COUNTER
    with CRON_LOCK:
        CRON_COUNTER += 1
        return f"cron-{CRON_COUNTER}"

#生成bg_id
def make_background_id()->str:
    global BACKGROUND_COUNTER
    #with只允许一个线程进入
    with BACKGROUND_LOCK:
        BACKGROUND_COUNTER += 1
        return f"bg-{BACKGROUND_COUNTER}"

#生成任务
def start_background_task(tool_call_id: str, tool_name: str, tool_args: dict) -> str:
    bg_id = make_background_id()
    command = tool_args.get("command", tool_name)
    #创建
    with BACKGROUND_LOCK:
        BACKGROUND_TASKS[bg_id] = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "args": tool_args,
            "command": command,
            "status": "running",
            "started_at": int(time.time())
        }
    #多线程函数
    def worker():
        _, handlers = assemble_tool_pool()
        handler = handlers.get(tool_name)
        if handler is None:
            output = f"Error: Unknown tool {tool_name}"
        else:
            try:
                output = handler(**tool_args)
            except Exception as e:
                output = f"Error: {e}"
        #运行完了再修改task状态
        with BACKGROUND_LOCK:
            BACKGROUND_TASKS[bg_id]["status"] = "completed"
            BACKGROUND_TASKS[bg_id]["completed_at"] = int(time.time())
            BACKGROUND_RESULTS[bg_id] = str(output)
    #开启多线程
    thread = threading.Thread(target= worker, daemon= True)
    thread.start()
    return bg_id

#收集完成的task
def collect_background_results() -> list:
    #已完成的task
    with BACKGROUND_LOCK:
        ready_ids = [
            bg_id
            for bg_id, task in BACKGROUND_TASKS.items()
            if task["status"] == "completed"
        ]

    notifications = []
    #对已完成的进行总结
    for bg_id in ready_ids:
        with BACKGROUND_LOCK:
            task = BACKGROUND_TASKS.pop(bg_id)
            output = BACKGROUND_RESULTS.pop(bg_id, "")

        summary = output[:200]
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>"
        )

    return notifications

#用来判断无工具调用时是否可以自动把in_progress变成completed
def has_running_background_tasks() -> bool:
    with BACKGROUND_LOCK:
        return any(task["status"] == "running" for task in BACKGROUND_TASKS.values())

#对比函数，用来处理时间的输入，对比
def _cron_field_matches(field: str, value: int) -> bool:
    if field == "*":
        return True

    for part in field.split(","):
        part = part.strip()

        if part.startswith("*/"):
            step = int(part[2:])
            if step <= 0:
                return False
            if value % step == 0:
                return True
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start <= value <= end:
                return True
            continue

        if int(part) == value:
            return True

    return False

#比较当前时间是否匹配输入时间
def cron_matches(cron_expr: str, dt: datetime) -> bool:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False

    minute, hour, dom, month, dow = fields
    dow_value = (dt.weekday() + 1) % 7

    minute_ok = _cron_field_matches(minute, dt.minute)
    hour_ok = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_value)

    if not (minute_ok and hour_ok and month_ok):
        return False

    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"

    if dom_unconstrained and dow_unconstrained:
        return True
    if dom_unconstrained:
        return dow_ok
    if dow_unconstrained:
        return dom_ok

    return dom_ok or dow_ok

#检查输入的单个时间参数是否合法
def _validate_cron_field(field: str, low: int, high: int) -> str:
    if not field:
        return "empty field"

    for part in field.split(","):
        part = part.strip()
        if not part:
            return "empty list item"

        if part == "*":
            continue

        if part.startswith("*/"):
            step_text = part[2:]
            if not step_text.isdigit():
                return f"invalid step: {part}"
            step = int(step_text)
            if step <= 0:
                return f"invalid step: {part}"
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                return f"invalid range: {part}"
            start = int(start_text)
            end = int(end_text)
            if start > end:
                return f"invalid range: {part}"
            if start < low or end > high:
                return f"out of range: {part}"
            continue

        if not part.isdigit():
            return f"invalid value: {part}"

        value = int(part)
        if value < low or value > high:
            return f"out of range: {part}"

    return ""

#检查总体的时间参数是否合法
def validate_cron(cron_expr: str) -> str:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return "cron must have 5 fields: minute hour day month weekday"

    minute, hour, dom, month, dow = fields

    checks = [
        ("minute", minute, 0, 59),
        ("hour", hour, 0, 23),
        ("day", dom, 1, 31),
        ("month", month, 1, 12),
        ("weekday", dow, 0, 6),
    ]

    for name, field, low, high in checks:
        error = _validate_cron_field(field, low, high)
        if error:
            return f"{name}: {error}"

    return ""

#创建定时任务
def schedule_job(cron: str, prompt: str, recurring: bool = True) -> str:
    #如果不合法
    error = validate_cron(cron)
    if error:
        return f"Error: invalid cron: {error}"
    #创建基础信息
    job_id = make_cron_id()
    now = int(time.time())
    #创建job
    job = {
        "id": job_id,
        "cron": cron,
        "prompt": prompt,#触发时交给llm的提示词
        "recurring": recurring,#是否重复触发
        "created_at": now,
    }
    #加进定时任务列表
    with CRON_LOCK:
        SCHEDULED_JOBS[job_id] = job

    return f"Scheduled {job_id}: {cron} -> {prompt}"

#删除已经完成或者不需要的定时任务
def cancel_job(job_id: str) -> str:
    with CRON_LOCK:
        job = SCHEDULED_JOBS.pop(job_id, None)
        LAST_FIRED.pop(job_id, None)

    if not job:
        return f"Error: cron job not found: {job_id}"

    return f"Cancelled cron job {job_id}"

#定时任务的主体
def cron_scheduler_loop() -> None:
    while True:
        #一直检查是否到了时间
        time.sleep(1)
        now = datetime.now()
        minute_marker = now.strftime("%Y-%m-%d %H:%M")

        with CRON_LOCK:
            jobs = list(SCHEDULED_JOBS.values())

        for job in jobs:
            try:
                #没到时间
                if not cron_matches(job["cron"], now):
                    continue

                with CRON_LOCK:
                    if LAST_FIRED.get(job["id"]) == minute_marker:
                        continue
                    #不直接进行，只入队
                    CRON_QUEUE.append(job.copy())
                    LAST_FIRED[job["id"]] = minute_marker
                    #一次性任务直接删除
                    if not job.get("recurring", True):
                        SCHEDULED_JOBS.pop(job["id"], None)

            except Exception as e:
                print(f"[cron error] {job.get('id', '?')}: {e}")

#加入队列后需要触发行为，给llm取出来触发的函数
def consume_cron_queue() -> list:
    with CRON_LOCK:
        jobs = list(CRON_QUEUE)
        #每一轮的触发任务不一样，取出后就清空
        CRON_QUEUE.clear()

    return jobs

#task id 生成, 防止同一时间创建多个相同的id
def make_task_id()->str:
    return f"task-{int(time.time())}-{random.randint(0, 9999):04d}"

#用任务描述创建任务, 同时加上owner，可以让其他agent自动认领任务
def create_task(
        description: str,
        status: str = "in_progress",
        owner: str|None = None,
        blockedBy: list | None = None
):
    global CURRENT_TASK_ID

    task_id = make_task_id()
    now = int(time.time())

    #创建任务
    task = {
        "id": task_id,
        "description": description,
        "status": status,
        "owner": owner,
        "blockedBy": blockedBy or [],
        "worktree": None,
        "created_at": now,
        "updated_at": now,
        "result": ""
    }
    if status == "in_progress":
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

#自动agent认领任务判断是否可以认领，是否已经完成
def can_start(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return False
    for dep_id in task.get("blockedBy", []):
        dep = TASKS.get(dep_id)
        if not dep:
            return False
        if dep.get("status") != "completed":
            return False
    return True

#扫描任务板，看什么可以做
def scan_unclaimed_tasks():
    unclaimed = []

    for task in TASKS.values():
        if (
            task.get("status") == "pending"
            and not task.get("owner")
            and can_start(task["id"])
        ):
            unclaimed.append(task)
    return unclaimed

#teammate agent空闲时轮询
def idle_poll(agent_name: str, messages: list, role: str):
    for _ in range(IDLE_TIMEOUT // IDLE_POLL_INTERVAL):
        time.sleep(IDLE_POLL_INTERVAL)

        #优先检查inbox，lead的消息比任务板重要
        inbox = read_inbox(agent_name)
        if inbox:
            for msg in inbox:
                if msg.get("type") == "shutdown_request":
                    request_id = msg.get("metadata", {}).get("request_id", "")
                    send_message(
                        agent_name,
                        "lead",
                        "Shutting down gracefully.",
                        "shutdown_response",
                        {
                            "request_id": request_id, "approve": True
                        }
                    )
                    return "shutdown"

            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, ensure_ascii=False)}</inbox>"
            })
            return "work"

        #没有inbox任务，扫描任务板准备自己接任务
        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            task = unclaimed[0]
            result = claim_task(task["id"], owner= agent_name)
            #如果成功修改状态，认领成功
            if "Claimed" in result:
                messages.append({
                    "role": "user",
                    "content": (
                        f"<auto-claimed>"
                        f"Task {task['id']}: {task['description']}"
                        f"</auto-claimed>"
                    )
                })
                return "work"
    return "timeout"

#自动agent认领某个任务
def claim_task(task_id: str, owner: str = "agent"):
    task = TASKS.get(task_id)
    #如果不存在任务直接返回
    if not task:
        return f"Error: task not found: {task_id}"
    #如果不是代办状态直接返回
    if task.get("status") != "pending":
        return f"Task {task_id} is {task.get('status')}, cannot claim"
    #如果已经被认领
    if task.get("owner"):
        return f"Task {task_id} already owned by {task.get('owner')}"
    #前置任务没有完成
    if not can_start(task_id):
        return f"Task {task_id} is blocked"

    task["owner"] = owner
    task["status"] = "in_progress"
    task["updated_at"] = int(time.time())

    return f"Claimed {task_id}: {task.get('description')}"

#完成任务时修改状态
def complete_task(task_id: str, result: str = ""):
    task = TASKS.get(task_id)
    #如果没有这个任务
    if not task:
        return f"Error: task not found: {task_id}"
    #如果不是处理中的任务，不能更改为完成
    if task.get("status") != "in_progress":
        return f"Task {task_id} is {task.get('status')}, cannot complete"
    #更改任务状态
    task["status"] = "completed"
    task["updated_at"] = int(time.time())

    if result:
        task["result"] = result
    return f"Completed {task_id}: {task.get('description')}"

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
    _, handlers = assemble_tool_pool()
    enabled_tools = ",".join(handlers.keys())
    available_mcp_servers = ", ".join(MOCK_SERVERS.keys())
    connected_mcp_servers = ", ".join(MCP_CLIENTS.keys()) or "(none)"
    mcp_tool_names = ",".join(
        name
        for name in handlers.keys()
        if name.startswith("mcp__")
    ) or "(none)"
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
            "available_mcp_servers": available_mcp_servers,
            "connected_mcp_servers": connected_mcp_servers,
            "mcp_tool_names": mcp_tool_names,
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
    sections.append(
        "MCP external tools:\n"
        f"- Available mock MCP servers: {available_mcp_servers}\n"
        f"- Connected MCP servers: {connected_mcp_servers}\n"
        f"- Connected MCP tool names: {mcp_tool_names}\n"
        "- Use connect_mcp(name) to connect a server before using its tools.\n"
        "- MCP tools are named mcp__{server}__{tool}."
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
BUILTIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "run_in_background": {"type": "boolean"}},
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
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a pending task on the shared task board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "blockedBy": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List all tasks on the shared task board.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claim_task",
            "description": "Claim a pending task from the shared task board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Complete an in-progress task on the shared task board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "result": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_worktree",
            "description": "Create an isolated git worktree and optionally bind it to a task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "task_id": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_cron",
            "description": "Schedule a prompt to run later or repeatedly using a five-field cron expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cron": {"type": "string"},
                    "prompt": {"type": "string"},
                    "recurring": {"type": "boolean"},
                },
                "required": ["cron", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_crons",
            "description": "List all scheduled cron jobs.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_cron",
            "description": "Cancel a scheduled cron job by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message from the lead agent to a teammate inbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["to", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inbox",
            "description": "Read and consume the lead agent inbox.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_teammate",
            "description": "Start a teammate agent thread with a name, role, and task prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["name", "role", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_teammates",
            "description": "List teammate agents and their current status.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_shutdown",
            "description": "Request a teammate to shut down gracefully using a request/response protocol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "teammate": {"type": "string"},
                },
                "required": ["teammate"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_plan",
            "description": "Ask a teammate to submit a plan for a task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "teammate": {"type": "string"},
                    "task": {"type": "string"},
                },
                "required": ["teammate", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_plan",
            "description": "Approve or reject a teammate plan_approval request by request_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    "approve": {"type": "boolean"},
                    "feedback": {"type": "string"},
                },
                "required": ["request_id", "approve"],
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


BUILTIN_TOOLS.append({
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


BUILTIN_TOOLS.append({
    "type": "function",
    "function": {
        "name": "compact",
        "description": "Compact conversation history when context is getting long or noisy.",
        "parameters": {
            "type": "object",
            "properties": {},
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
def safe_path(p: str, cwd: Path | None = None) -> Path:
    # 默认在主工作目录；teammate 进入 worktree 后可以传 cwd，实现目录隔离。
    base = (cwd or WORKDIR).resolve()
    #拼出路径
    path = (base / p).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

#阅读文件
def run_read(path: str, limit: Optional[int] = None, cwd: Path | None = None) -> str:
    try:
        #用safe path保证路径正确
        lines = safe_path(path, cwd=cwd).read_text(encoding="utf-8").splitlines()
        #限制取前limit行
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

#写文件
def run_write(path: str, content: str, cwd: Path | None = None) -> str:
    try:
        file_path = safe_path(path, cwd=cwd)
        #没有父文件的话一起创建父文件，文件已存在不报错
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
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

def run_bash(command: str, run_in_background: bool = False, cwd: Path | None = None):
    #异常拦截
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    #开始正式运行command,.run表示运行系统命令
    try:
        r = subprocess.run(
            command,
            shell= True,#用shell执行命令
            cwd= str((cwd or WORKDIR).resolve()),#默认主目录；teammate 可传 worktree 目录
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

#多agent协作，lead agent专用的write和read工具
def run_send_message(to: str, content: str) -> str:
    return send_message("lead", to, content)


def run_check_inbox() -> str:
    messages = consume_lead_inbox(route_protocol=True)
    if not messages:
        return "Inbox is empty."

    return json.dumps(messages, ensure_ascii=False, indent=2)

#teammate agent的tools
TEAMMATE_TOOLS = [
    BUILTIN_TOOLS[0],  # bash
    BUILTIN_TOOLS[1],  # read_file
    BUILTIN_TOOLS[2],  # write_file
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to the lead agent or another teammate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["to", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit a plan to the lead agent for approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string"},
                },
                "required": ["plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List all tasks on the board.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claim_task",
            "description": "Claim a pending task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark an in-progress task as completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "result": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
    },
]

#多agent协作，lead agent创建teammate agent，开线程
def spawn_teammate_thread(name:str, role: str, prompt: str)-> str:
    if name in ACTIVE_TEAMMATES:
        return f"Error: teammate already exists: {name}"

    def handle_inbox_message(msg: dict, messages: list) -> bool:
        # 协议消息不直接丢给 LLM 自由理解，而是先由程序按 type 分流处理。
        msg_type = msg.get("type", "message")
        metadata = msg.get("metadata", {})
        request_id = metadata.get("request_id", "")

        if msg_type == "shutdown_request":
            send_message(
                name,
                "lead",
                "Shutting down gracefully.",
                "shutdown_response",
                {"request_id": request_id, "approve": True},
            )
            return True

        if msg_type == "plan_approval_response":
            approve = metadata.get("approve", False)
            if approve:
                messages.append({
                    "role": "user",
                    "content": "[Plan approved] Proceed with the task.",
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"[Plan rejected] Feedback: {msg.get('content', '')}",
                })

        return False

    #线程函数,给lead发消息
    def worker():
        try:
            wt_ctx = {"path": None}

            def current_workdir():
                if wt_ctx["path"]:
                    return Path(wt_ctx["path"])
                return None

            def teammate_bash(command: str, run_in_background: bool = False) -> str:
                return run_bash(
                    command,
                    run_in_background=run_in_background,
                    cwd=current_workdir(),
                )

            def teammate_read_file(path: str, limit: int | None = None) -> str:
                return run_read(path, limit=limit, cwd=current_workdir())

            def teammate_write_file(path: str, content: str) -> str:
                return run_write(path, content, cwd=current_workdir())

            def teammate_send_message(to: str, content: str)->str:
                return send_message(name, to, content)

            def teammate_submit_plan(plan: str) -> str:
                return _teammate_submit_plan(name, plan)

            def teammate_list_tasks() -> str:
                if not TASKS:
                    return "No tasks."
                return json.dumps(list(TASKS.values()), ensure_ascii=False, indent=2)

            def teammate_claim_task(task_id: str) -> str:
                result = claim_task(task_id, owner=name)

                if "Claimed" in result:
                    task = TASKS.get(task_id)
                    if task and task.get("worktree"):
                        wt_ctx["path"] = str(WORKTREES_DIR / task["worktree"])

                return result

            def teammate_complete_task(task_id: str, result: str = "") -> str:
                return complete_task(task_id, result)

            #给teammate agent用的函数映射
            teammate_handlers = {
                "bash": teammate_bash,
                "read_file": teammate_read_file,
                "write_file": teammate_write_file,
                "send_message": teammate_send_message,
                "submit_plan": teammate_submit_plan,
                "list_tasks": teammate_list_tasks,
                "claim_task": teammate_claim_task,
                "complete_task": teammate_complete_task,
            }
            #创建teammate初始提示词
            system = (
                f"You are '{name}', a {role}. "
                f"You are a teammate agent working at {WORKDIR}. "
                "Use tools to complete the task. "
                "Check inbox messages for protocol requests. "
                "If you need approval before risky work, use submit_plan. "
                "When finished, send a concise result to 'lead' using send_message."
            )
            #自己的对话流程记忆，短期记忆
            messages = [{"role": "user", "content": prompt}]
            shutdown_requested = False

            while not shutdown_requested:
                #重复多次自动压缩messages后，需要重新注入身份
                if len(messages) <= 3:
                    messages.insert(0,{
                        "role": "user",
                        "content": (
                            f"<identity>You are '{name}', role: {role}. "
                            f"Continue your work.</identity>"
                        )
                    })
                #读取是否有新消息；协议消息先分流，普通消息再交给 LLM
                inbox = read_inbox(name)
                non_protocol_messages = []
                for msg in inbox:
                    if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                        shutdown_requested = handle_inbox_message(msg, messages)
                        if shutdown_requested:
                            break
                    else:
                        non_protocol_messages.append(msg)

                if shutdown_requested:
                    break

                if non_protocol_messages:
                    messages.append({
                        "role": "user",
                        "content": f"<inbox>{json.dumps(non_protocol_messages, ensure_ascii=False)}</inbox>"
                    })
                #调用llm
                response = with_retry(
                    lambda: client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": system}, *messages],
                        tools=TEAMMATE_TOOLS,
                        max_tokens=DEFAULT_MAX_TOKENS,
                    )
                )
                #提取信息
                message = response.choices[0].message
                assistant_message = message.model_dump(exclude_none=True)
                messages.append(assistant_message)

                if not message.tool_calls:
                    # 没有工具调用时不立刻退出，而是进入 idle，等待 Lead 后续发协议消息或新任务。
                    ACTIVE_TEAMMATES[name]["status"] = "idle"
                    idle_result = idle_poll(name, messages, role)

                    if idle_result == "work":
                        ACTIVE_TEAMMATES[name]["status"] = "running"
                        continue

                    if idle_result == "shutdown":
                        shutdown_requested = True
                        break

                    if idle_result == "timeout":
                        break


                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        output = f"Error: invalid JSON arguments for {tool_name}: {e}"
                        messages.append(make_tool_result_message(tool_call.id, output))
                        continue

                    handler = teammate_handlers.get(tool_name)
                    if handler is None:
                        output = f"Error: Unknown tool {tool_name}"
                    else:
                        output = handler(**tool_args)

                    messages.append(make_tool_result_message(tool_call.id, output))
            #运行完后总结
            summary = "Done."
            #只总结运行调用的结果
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    summary = extract_text(msg) or summary
                    break
            #发送消息给lead
            send_message(name, "lead", summary, "result")
            #标志为完成
            ACTIVE_TEAMMATES[name]["status"] = "completed"
            ACTIVE_TEAMMATES[name]["completed_at"] = int(time.time())
        except Exception as e:
            ACTIVE_TEAMMATES[name]["status"] = "failed"
            ACTIVE_TEAMMATES[name]["error"] = str(e)
            ACTIVE_TEAMMATES[name]["completed_at"] = int(time.time())
            send_message(name, "lead", f"Teammate failed: {e}", "error")

    thread = threading.Thread(target= worker, daemon= True)
    ACTIVE_TEAMMATES[name] = {
        "name": name,
        "role": role,
        "status": "running",
        "started_at": int(time.time()),
        "thread": thread
    }

    thread.start()
    return f"Spawned teammate {name} as {role}"


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    # teammate 发起计划审批：请求登记在同一本 PENDING_REQUESTS 账本里。
    request_id = make_request_id()
    PENDING_REQUESTS[request_id] = ProtocolState(
        request_id=request_id,
        type="plan_approval",
        sender=from_name,
        target="lead",
        status="pending",
        payload=plan,
    )
    send_message(
        from_name,
        "lead",
        plan,
        "plan_approval_request",
        {"request_id": request_id},
    )
    return f"Plan submitted ({request_id}). Waiting for approval."


def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    return spawn_teammate_thread(name, role, prompt)


# 给 lead agent 查看 teammate 线程状态；这和 check_inbox 不同，inbox 看消息，这里看运行状态。
def run_list_teammates() -> str:
    if not ACTIVE_TEAMMATES:
        return "No active teammates."

    teammates = []
    for name, teammate in ACTIVE_TEAMMATES.items():
        item = {
            "name": name,
            "role": teammate.get("role"),
            "status": teammate.get("status"),
            "started_at": teammate.get("started_at"),
            "completed_at": teammate.get("completed_at"),
            "error": teammate.get("error"),
        }
        teammates.append(item)

    return json.dumps(teammates, ensure_ascii=False, indent=2)


def run_request_shutdown(teammate: str) -> str:
    request_id = make_request_id()
    PENDING_REQUESTS[request_id] = ProtocolState(
        request_id=request_id,
        type="shutdown",
        sender="lead",
        target=teammate,
        status="pending",
        payload="Please shut down gracefully.",
    )
    send_message(
        "lead",
        teammate,
        "Please shut down gracefully.",
        "shutdown_request",
        {"request_id": request_id},
    )
    return f"Shutdown request sent to {teammate} ({request_id})"


def run_request_plan(teammate: str, task: str) -> str:
    # 这是普通消息触发 teammate 自己调用 submit_plan；真正的审批请求由 teammate 发起。
    return send_message(
        "lead",
        teammate,
        f"Please submit a plan for: {task}",
        "message",
    )


def run_review_plan(request_id: str, approve: bool, feedback: str = "") -> str:
    state = PENDING_REQUESTS.get(request_id)
    if not state:
        return f"Error: unknown request_id: {request_id}"

    if state.type != "plan_approval":
        return f"Error: request {request_id} is {state.type}, not plan_approval"

    if state.status != "pending":
        return f"Request {request_id} already {state.status}"

    state.status = "approved" if approve else "rejected"
    send_message(
        "lead",
        state.sender,
        feedback or ("Approved" if approve else "Rejected"),
        "plan_approval_response",
        {"request_id": request_id, "approve": approve},
    )
    return f"Plan {state.status} ({request_id})"

#判断是否是慢指令
def is_slow_operation(tool_name: str, tool_args: dict)->bool:
    #不是bash命令运行的
    if tool_name != "bash":
        return False
    #取出命令
    command = tool_args.get("command", "").lower()
    #需要background的命令
    slow_keywords = [
        "install", "build", "test", "deploy", "compile",
        "docker build", "pip install", "npm install",
        "cargo build", "pytest", "make",
    ]
    return any(keyword in command for keyword in slow_keywords)

#判断是否需要background run, harness的判断作为最后的防线，第一顺位是llm输入run_in_background
def should_run_background(tool_name: str, tool_args: dict)->bool:
    if tool_args.get("run_in_background"):
        return True
    return is_slow_operation(tool_name= tool_name, tool_args= tool_args)


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


def run_create_task(description: str, blockedBy: list | None = None) -> str:
    task = create_task(
        description,
        status="pending",
        owner=None,
        blockedBy=blockedBy,
    )
    return f"Created {task['id']}: {task['description']}"


def run_list_tasks() -> str:
    if not TASKS:
        return "No tasks."
    return json.dumps(list(TASKS.values()), ensure_ascii=False, indent=2)


def run_claim_task(task_id: str) -> str:
    return claim_task(task_id, owner="lead")


def run_complete_task(task_id: str, result: str = "") -> str:
    return complete_task(task_id, result)


def run_create_worktree(name: str, task_id: str = "") -> str:
    return create_worktree(name, task_id)


def run_schedule_cron(cron: str, prompt: str, recurring: bool = True) -> str:
    return schedule_job(cron, prompt, recurring)


def run_list_crons() -> str:
    with CRON_LOCK:
        jobs = list(SCHEDULED_JOBS.values())

    if not jobs:
        return "No scheduled cron jobs."

    return json.dumps(jobs, ensure_ascii=False, indent=2)


def run_cancel_cron(job_id: str) -> str:
    return cancel_job(job_id)

#str name对应到具体的执行函数
BUILTIN_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_write": run_todo_write,
    "load_skill": load_skill,
    "task_status": run_task_status,
    "task_update": run_task_update,
    "create_task": run_create_task,
    "list_tasks": run_list_tasks,
    "claim_task": run_claim_task,
    "complete_task": run_complete_task,
    "create_worktree": run_create_worktree,
    "schedule_cron": run_schedule_cron,
    "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
    "send_message": run_send_message,
    "check_inbox": run_check_inbox,
    "spawn_teammate": run_spawn_teammate,
    "list_teammates": run_list_teammates,
    "request_shutdown": run_request_shutdown,
    "request_plan": run_request_plan,
    "review_plan": run_review_plan,
}

#某个外部服务的本地代理。MCP
class MCPClient:
    def __init__(self, name: str):
        self.name = name
        self.tools = []
        self._handlers = {}

    def register(self, tool_defs: list, handlers: dict):
        self.tools = tool_defs
        self._handlers = handlers

    def call_tool(self, tool_name: str, args: dict):
        handler = self._handlers.get(tool_name)
        if handler is None:
            return f"MCP error: unknown tool '{tool_name}'"

        try:
            return handler(**args)
        except Exception as e:
            return f"MCP error: {e}"

MCP_CLIENTS = {}

DISALLOWED_MCP_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_mcp_name(name: str) -> str:
    return DISALLOWED_MCP_NAME_CHARS.sub("_", name)

#模拟两个mcp server, 都通过mcpclient来注册工具
#docs server
def _mock_server_docs():
    client = MCPClient("docs")
    client.register(
        tool_defs=[
            {
                "name": "search",
                "description": "Search documentation. (readOnly)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_version",
                "description": "Get API version. (readOnly)",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ],
        handlers={
            "search": lambda query: f"[docs] Found 3 results for '{query}'",
            "get_version": lambda: "[docs] API v2.1.0",
        },
    )
    return client

#deploy server
def _mock_server_deploy():
    client = MCPClient("deploy")
    client.register(
        tool_defs=[
            {
                "name": "trigger",
                "description": "Trigger a deployment. (destructive)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                    },
                    "required": ["service"],
                },
            },
            {
                "name": "status",
                "description": "Check deployment status. (readOnly)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                    },
                    "required": ["service"],
                },
            },
        ],
        handlers={
            "trigger": lambda service: f"[deploy] Triggered: {service}",
            "status": lambda service: f"[deploy] {service}: running (v1.4.2)",
        },
    )
    return client

#只是注册存在这些外部服务，不是已经注册成为mcpclient
MOCK_SERVERS = {
    "docs": _mock_server_docs,
    "deploy": _mock_server_deploy,
}

#将存在的外部服务注册在全局环境内，成为一个mcpclient
def connect_mcp(name: str) -> str:
    #已经注册完了
    if name in MCP_CLIENTS:
        return f"MCP server '{name}' already connected"
    factory = MOCK_SERVERS.get(name)
    if not factory:
        available = ", ".join(MOCK_SERVERS.keys())
        return f"Unknown server '{name}'. Available: {available}"

    mcp_client = factory()
    MCP_CLIENTS[name] = mcp_client

    tool_names = [tool["name"] for tool in mcp_client.tools]
    tool_list = ", ".join(tool_names)
    return (
        f"Connected to MCP server '{name}'. "
        f"Discovered {len(tool_names)} tools: {tool_list}"
    )


def run_connect_mcp(name: str) -> str:
    return connect_mcp(name)


BUILTIN_TOOLS.append({
    "type": "function",
    "function": {
        "name": "connect_mcp",
        "description": (
            "Connect to a mock MCP server by name. "
            "Available servers: docs, deploy. "
            "After connecting, the server's tools become available in later turns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
    },
})
BUILTIN_HANDLERS["connect_mcp"] = run_connect_mcp


def make_mcp_handler(mcp_client: MCPClient, tool_name: str):
    def handler(**kwargs):
        return mcp_client.call_tool(tool_name, kwargs)

    return handler


#返回内置tools，handler和外部的tools和handler。TOOLS和HANDLER作为内置的tool
def assemble_tool_pool():
    tools = list(BUILTIN_TOOLS)
    handlers = dict(BUILTIN_HANDLERS)

    for server_name, mcp_client in MCP_CLIENTS.items():
        safe_server = normalize_mcp_name(server_name)

        for tool_def in mcp_client.tools:
            original_tool_name = tool_def["name"]
            safe_tool = normalize_mcp_name(original_tool_name)
            prefixed_name = f"mcp__{safe_server}__{safe_tool}"

            tools.append({
                "type": "function",
                "function": {
                    "name": prefixed_name,
                    "description": tool_def.get("description", ""),
                    "parameters": tool_def.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                    }),
                },
            })
            handlers[prefixed_name] = make_mcp_handler(
                mcp_client,
                original_tool_name,
            )

    return tools, handlers

#sub agent的可用工具，不能再给task，因为可能会无限递归
SUB_TOOLS = BUILTIN_TOOLS[:5]

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
    if tool_name.startswith("mcp__") and "__deploy__" in tool_name:
        return f"MCP deploy tool may change external state: {tool_name}"
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


BUILTIN_TOOLS.append({
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
BUILTIN_HANDLERS["task"] = spawn_subagent

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
        #检查是否有定时任务
        #已经在等待队列的任务
        fired_jobs = consume_cron_queue()
        for job in fired_jobs:
            messages.append(
                {
                    "role": "user",
                    "content": f"[Scheduled] {job['prompt']}"
                }
            )
        #检查后台有没有线程完成
        bg_notifications = collect_background_results()
        if bg_notifications:
            messages.append(
                {
                    "role": "user",
                    "content": "\n\n".join(bg_notifications)
                }
            )

        # 检查是否有 teammate agent 的新消息；协议响应要先路由更新 PENDING_REQUESTS。
        lead_inbox = consume_lead_inbox(route_protocol=True)
        if lead_inbox:
            inbox_text = "\n".join(
                (
                    f"From {msg.get('from')} "
                    f"[type={msg.get('type')}, request_id={msg.get('metadata', {}).get('request_id', '')}]: "
                    f"{msg.get('content')}"
                )
                for msg in lead_inbox
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"[Inbox]\n{inbox_text}"
                }
            )
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
        tools, handlers = assemble_tool_pool()
        try:
            #连接llm获得回答
            response = with_retry(
                lambda: client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": build_system(memories=memories)}, *messages],
                    tools=tools,
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
            if current_task and current_task["status"] == "in_progress" and not has_running_background_tasks():
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

            if tool_name == "compact":
                messages[:] = compact_history(messages)
                output = "Compacted conversation history. Continue with the summarized context."
                messages.append(make_tool_result_message(tool_call_id=tool_call.id, output=output))
                continue

            #用hook来走pretooluse流程
            blocked = trigger_hooks("PreToolUse", tool_name, tool_args)
            if blocked:
                messages.append(make_tool_result_message(tool_call_id= tool_call.id, output= str(blocked)))
                continue

            #得到具体函数
            handler = handlers.get(tool_name)

            if handler is None:
                logger.warning("unknown tool requested: %s", tool_call.function.name)
                output = f"Error: Unknown tool {tool_call.function.name}"
            #判断是否需要作为后台任务异步进行
            elif should_run_background(tool_name, tool_args):
                bg_id = start_background_task(tool_call.id, tool_name, tool_args)
                output = (
                    f"[Background task {bg_id} started] "
                    "Result will be available when complete."
                )

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
    #启动时创建线程来进行定时任务的监测
    cron_thread = threading.Thread(target= cron_scheduler_loop, daemon= True)
    cron_thread.start()
    print("cron线程开启")

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
