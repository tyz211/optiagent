from __future__ import annotations

from functools import wraps
from threading import Lock
from weakref import WeakValueDictionary

from fastapi import HTTPException


# 默认单服务进程内，每个会话同时只处理一轮；弱引用避免闲置会话锁长期占内存。
_locks = WeakValueDictionary()
_registry_lock = Lock()


def serialize_conversation(function):
    """拒绝同一会话的重叠写入，防止两轮同时以旧版本为基础覆盖对方。"""
    @wraps(function)
    def guarded(*args, **kwargs):
        conversation_id = kwargs.get('conversation_id')
        if conversation_id is None:
            return function(*args, **kwargs)
        key = (kwargs.get('user_id'), conversation_id)
        with _registry_lock:
            lock = _locks.get(key)
            if lock is None:
                lock = Lock()
                _locks[key] = lock
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail='当前对话仍在处理上一条消息，请完成后再发送。')
        try:
            return function(*args, **kwargs)
        finally:
            lock.release()
    return guarded
