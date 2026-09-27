from __future__ import annotations

import asyncio
import anyio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import BoundedSemaphore
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'OPTIAGENT_DEMO '
TIMEOUT_SECONDS = 90
# 全部全局故障补丁都仅存在于子进程；当前 API 进程只接收结构化事件。
SLOTS = BoundedSemaphore(2)
router = APIRouter()
TEMPLATES = {'knapsack': '背包选择', 'assignment': '资源指派', 'tsp': '路线规划',
             'job_shop_scheduling': '车间调度', 'production_mix': '产品组合', 'facility_location': '仓库选址'}
SCENARIOS = {
    'clean': ('正常输入', '输入正确，完成求解并独立验算。'),
    'transient_failure': ('短暂传输失败', '第一次未到达求解器，重试可恢复。'),
    'tampered_objective': ('短暂映射错误', '第一次收到错误系数，后续可取得正确输入。'),
    'model_error': ('持续映射错误', '重试保留错误映射，必须重新绑定才能恢复。'),
    'persistent_failure': ('无法修复的映射', '重建后仍无法取得正确映射，应拒绝接受错误解。'),
}


class RepairDemoRequest(BaseModel):
    """只接受有限的演示选项，客户端不能提交脚本、模型路径或数据库路径。"""
    model_config = ConfigDict(extra='forbid')
    template_id: Literal['knapsack', 'assignment', 'tsp', 'job_shop_scheduling', 'production_mix', 'facility_location'] = 'knapsack'
    scenario: Literal['clean', 'transient_failure', 'tampered_objective', 'model_error', 'persistent_failure'] = 'model_error'
    policy: Literal['rule', 'previous', 'learned'] = 'learned'
    seed: int = Field(default=7300, ge=0, le=1000000, strict=True)
    instance_index: int = Field(default=0, ge=0, le=99, strict=True)


def checkpoints() -> dict[str, str]:
    """检查点只从服务器配置读取；默认使用已验收的固定种子，绝不按当前结果挑模型。"""
    return {
        'rule': '',
        'previous': os.environ.get('OPTIAGENT_DEMO_PREVIOUS_CHECKPOINT', str(ROOT / 'artifacts/rl/studies/20260919-five-seed-v2/runs/seed-11/recovery_policy.pt')),
        'learned': os.environ.get('OPTIAGENT_DEMO_CHECKPOINT', str(ROOT / 'artifacts/rl/studies/20260921-repair-demo/runs/seed-11/recovery_policy.pt')),
    }


def policy_available(policy: str) -> bool:
    """规则不依赖 PyTorch，学习策略必须明确有可读取的本地权重。"""
    return policy == 'rule' or (bool(checkpoints()[policy]) and Path(checkpoints()[policy]).is_file() and importlib.util.find_spec('torch') is not None)


@router.get('/repair-demo')
def page():
    """与普通求解共享服务，但使用独立页面明确标注演示语义。"""
    return FileResponse(ROOT / 'web/repair-demo.html')


@router.get('/api/demo/repair/catalog')
def catalog():
    """返回界面选项与模型可用性，不向客户端公开模型的本地路径。"""
    return {'mode': 'live_controlled_demo', 'templates': TEMPLATES,
            'scenarios': {key: {'label': value[0], 'description': value[1]} for key, value in SCENARIOS.items()},
            'policies': [{'id': key, 'label': label, 'available': policy_available(key)}
                         for key, label in [('rule', '规则策略'), ('previous', '上一轮模型'), ('learned', '新修复模型')]],
            'timeout_seconds': TIMEOUT_SECONDS, 'source': '生成实例与受控故障；独立进程实时求解，不调用 LLM'}


def sse(kind: str, payload: dict) -> str:
    """只输出固定 JSON 事件，避免换行内容改变 SSE 消息边界。"""
    return f'event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n'


async def stream_run(request: RepairDemoRequest):
    """取消或超时都回收子进程及并发额度，普通对话数据库从不参与此调用。"""
    if not SLOTS.acquire(blocking=False):
        yield sse('error', {'message': '演示正在处理其他任务，请稍后重试。'})
        return
    process = None
    temporary = None
    try:
        run_id = uuid4().hex
        yield sse('started', {'run_id': run_id, 'mode': 'live_controlled_demo', 'request': request.model_dump()})
        temporary = tempfile.TemporaryDirectory(prefix='optiagent-demo-')
        environment = dict(os.environ, PYTHONUNBUFFERED='1', OPTIAGENT_SOLVER_THREADS='1', TMPDIR=temporary.name)
        async with asyncio.timeout(TIMEOUT_SECONDS):
            # 启动过程也可能收到断连；先取得进程句柄，确保 finally 能回收它。
            spawning = asyncio.create_task(asyncio.create_subprocess_exec(
                sys.executable, str(ROOT / 'scripts/run_live_repair_demo.py'), request.model_dump_json(), run_id,
                cwd=ROOT, env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                limit=1024 * 1024,
            ))
            try:
                process = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                with anyio.CancelScope(shield=True):
                    process = await spawning
                raise
            completed = False
            while line := await process.stdout.readline():
                text = line.decode('utf-8')
                if not text.startswith(PREFIX):
                    continue
                event = json.loads(text[len(PREFIX):])
                if event.get('event') not in {'input', 'node', 'evidence', 'final', 'error'} or not isinstance(event.get('data'), dict):
                    raise ValueError('演示进程事件合同错误。')
                yield sse(event['event'], event['data'])
                if event['event'] in {'final', 'error'}:
                    completed = True
                    break
            if not completed:
                yield sse('error', {'message': '演示进程未返回完整结果，请重试或检查服务日志。'})
    except TimeoutError:
        yield sse('error', {'message': '本次演示运行超时，进程已停止。请换一个实例重试。'})
    except (OSError, ValueError, RuntimeError):
        yield sse('error', {'message': '演示运行失败，请确认求解器和模型配置可用。'})
    finally:
        try:
            if process is not None:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                # Starlette 在断连时使用取消域，屏蔽清理阶段以等待已终止的子进程。
                with anyio.CancelScope(shield=True):
                    await process.wait()
        finally:
            # 即使客户端取消导致等待被打断，也清除该任务的临时数据并归还名额。
            if temporary is not None:
                temporary.cleanup()
            SLOTS.release()


@router.post('/api/demo/repair/stream')
async def run_demo(request: RepairDemoRequest):
    """显式的实时演示接口，不接受任意 checkpoint 或生产会话标识。"""
    if not policy_available(request.policy):
        raise HTTPException(status_code=503, detail='所选策略的检查点或 PyTorch 不可用。请先配置模型，或明确选择规则策略。')
    return StreamingResponse(stream_run(request), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
