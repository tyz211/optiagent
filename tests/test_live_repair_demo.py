from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from threading import BoundedSemaphore
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.services import repair_demo as demo


def decode_event(chunk):
    """按公开的 SSE 合同读取事件，避免测试依赖内部传输细节。"""
    lines = chunk.strip().splitlines()
    return lines[0].removeprefix('event: '), json.loads(lines[1].removeprefix('data: '))


class DemoRequestTests(unittest.TestCase):
    """演示参数受限；模型缺失不会悄悄使用规则。"""

    def setUp(self):
        app = FastAPI()
        app.include_router(demo.router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_catalog_and_page_without_checkpoint(self):
        with patch.dict('os.environ', {'OPTIAGENT_DEMO_CHECKPOINT': '/missing/model.pt'}):
            catalog = self.client.get('/api/demo/repair/catalog').json()
            self.assertEqual(6, len(catalog['templates']))
            self.assertEqual(5, len(catalog['scenarios']))
            self.assertFalse(next(p for p in catalog['policies'] if p['id'] == 'learned')['available'])
            self.assertNotIn('/missing', json.dumps(catalog))
            response = self.client.post('/api/demo/repair/stream', json={'policy': 'learned'})
            self.assertEqual(503, response.status_code)
        self.assertEqual(200, self.client.get('/repair-demo').status_code)

    def test_rejects_paths_unknown_scenarios_and_unbounded_instances(self):
        for payload in ({'checkpoint': '/tmp/arbitrary.pt'}, {'scenario': 'unknown'},
                        {'seed': -1}, {'seed': True}, {'seed': 1000001}, {'instance_index': 100}):
            with self.subTest(payload=payload):
                self.assertEqual(422, self.client.post('/api/demo/repair/stream', json=payload).status_code)


class DemoLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """验证提前关闭、超时、并发满额和启动中取消都不会遗留工作进程。"""

    def fake_process(self):
        process = unittest.mock.Mock(returncode=None)
        process.stdout.readline = AsyncMock(return_value=b'OPTIAGENT_DEMO {"event":"input","data":{}}\n')
        process.wait = AsyncMock(return_value=-9)
        return process

    async def test_close_reaps_process_and_temporary_data(self):
        process = self.fake_process()
        slot = BoundedSemaphore(1)
        with patch.object(demo, 'SLOTS', slot), patch.object(demo.asyncio, 'create_subprocess_exec', AsyncMock(return_value=process)) as spawn:
            stream = demo.stream_run(demo.RepairDemoRequest(policy='rule'))
            self.assertEqual('started', decode_event(await anext(stream))[0])
            self.assertEqual('input', decode_event(await anext(stream))[0])
            temporary = Path(spawn.call_args.kwargs['env']['TMPDIR'])
            self.assertTrue(temporary.exists())
            await stream.aclose()
            process.kill.assert_called_once()
            process.wait.assert_awaited_once()
            self.assertFalse(temporary.exists())
            self.assertTrue(slot.acquire(blocking=False))
            slot.release()

    async def test_timeout_reaps_process(self):
        process = self.fake_process()

        async def blocked():
            await asyncio.Event().wait()

        process.stdout.readline.side_effect = blocked
        slot = BoundedSemaphore(1)
        with patch.object(demo, 'SLOTS', slot), patch.object(demo, 'TIMEOUT_SECONDS', 0.02), patch.object(demo.asyncio, 'create_subprocess_exec', AsyncMock(return_value=process)):
            events = [decode_event(item) async for item in demo.stream_run(demo.RepairDemoRequest(policy='rule'))]
        self.assertEqual(['started', 'error'], [item[0] for item in events])
        self.assertIn('超时', events[-1][1]['message'])
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()
        self.assertTrue(slot.acquire(blocking=False))
        slot.release()

    async def test_cancel_during_spawn_still_reaps_child(self):
        process = self.fake_process()
        spawning = asyncio.Event()
        release = asyncio.Event()

        async def delayed_spawn(*args, **kwargs):
            spawning.set()
            await release.wait()
            return process

        with patch.object(demo.asyncio, 'create_subprocess_exec', delayed_spawn):
            stream = demo.stream_run(demo.RepairDemoRequest(policy='rule'))
            await anext(stream)
            pending = asyncio.create_task(anext(stream))
            await spawning.wait()
            pending.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()

    async def test_busy_does_not_start_worker(self):
        slot = BoundedSemaphore(1)
        slot.acquire()
        with patch.object(demo, 'SLOTS', slot), patch.object(demo.asyncio, 'create_subprocess_exec', AsyncMock()) as spawn:
            events = [decode_event(item) async for item in demo.stream_run(demo.RepairDemoRequest(policy='rule'))]
        self.assertEqual(['error'], [item[0] for item in events])
        spawn.assert_not_called()
        self.assertFalse(slot.acquire(blocking=False))
        slot.release()


class DemoIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """使用真实子进程与求解器并发执行不同故障，检查普通进程和数据库不被修改。"""

    async def test_concurrent_repair_and_safe_termination_are_isolated(self):
        from api import database
        from api.services import agent_workflow
        graph = agent_workflow.AGENT_GRAPH

        async def execute(scenario):
            return [decode_event(item) async for item in demo.stream_run(demo.RepairDemoRequest(policy='rule', scenario=scenario))]

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / 'normal.sqlite3'
            with patch.object(database, 'DB_PATH', database_path):
                database.init_db()
                before = database_path.read_bytes()
                repaired, stopped = await asyncio.gather(execute('model_error'), execute('persistent_failure'))
                self.assertEqual(before, database_path.read_bytes())
                self.assertEqual(database_path, database.DB_PATH)
        self.assertIs(graph, agent_workflow.AGENT_GRAPH)
        finals = []
        for events in (repaired, stopped):
            self.assertEqual('started', events[0][0])
            self.assertEqual('final', events[-1][0], events[-1])
            self.assertIn('node', [event[0] for event in events])
            evidence = [event[1] for event in events if event[0] == 'evidence']
            final = events[-1][1]
            self.assertEqual(evidence, final['case']['repair_events'])
            self.assertFalse(final['persisted_to_live_database'])
            self.assertFalse(final['llm_used'])
            self.assertFalse(final['case']['unverified_accept'])
            finals.append(final)
        self.assertTrue(finals[0]['case']['success'])
        self.assertIn('rebuild_model', finals[0]['case']['actions'])
        self.assertFalse(finals[1]['case']['success'])
        self.assertEqual('terminate', finals[1]['case']['actions'][-1])
        self.assertNotEqual(finals[0]['run_id'], finals[1]['run_id'])
        self.assertEqual(finals[0]['case']['instance_fingerprint'], finals[1]['case']['instance_fingerprint'])


if __name__ == '__main__':
    unittest.main()
