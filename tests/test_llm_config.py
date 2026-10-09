"""验证模型配置跨登录与重启持久化、空密钥保留和账户级 LLM 主控路由。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from fastapi.testclient import TestClient

from api import database
from api.main import app
from optiagent.llm import LLMConfig, list_openai_compatible_models


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"name": "custom", "base_url": "https://example.test/v1", "model": "saved-model",
          "api_key": "test-persistent-secret", "temperature": 0.4}


class LLMConfigTests(unittest.TestCase):
    """只在临时数据库保存合成凭据，不调用外部模型服务。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(database, "DB_PATH", Path(self.temp.name) / "settings.sqlite3")
        self.db_patch.start()
        self.client = TestClient(app)
        self.client.__enter__()
        self.headers = self.login("配置用户甲")

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.db_patch.stop()
        self.temp.cleanup()

    def login(self, username):
        """登录同一用户名应读取固定账户，不创建新的配置作用域。"""
        response = self.client.post("/api/login", json={"username": username})
        self.assertEqual(200, response.status_code)
        return {"X-Session-Token": response.json()["session_token"]}

    def save(self, config=None, headers=None):
        return self.client.post("/api/llm-config", headers=headers or self.headers,
                                json=CONFIG if config is None else config)

    def record(self, headers=None):
        """仅在测试内部核对原始凭据，公共 API 不返回密钥。"""
        token = (headers or self.headers)["X-Session-Token"]
        user = database.get_user_by_token(token)
        return database.get_active_llm_config(user["id"])

    def test_config_survives_restart_and_relogin_without_exposing_key(self):
        self.assertEqual(200, self.save().status_code)
        # 新客户端重新执行应用启动，并用同一用户名重新登录。
        with TestClient(app) as restarted:
            login = restarted.post("/api/login", json={"username": "配置用户甲"}).json()
            headers = {"X-Session-Token": login["session_token"]}
            response = restarted.get("/api/llm-config", headers=headers)
        self.assertEqual(self.headers, headers)
        self.assertTrue(response.json()["configured"])
        self.assertTrue(response.json()["has_api_key"])
        self.assertEqual(CONFIG["base_url"], response.json()["base_url"])
        self.assertEqual(CONFIG["model"], response.json()["model"])
        self.assertEqual(CONFIG["temperature"], response.json()["temperature"])
        self.assertNotIn("api_key", response.json())
        self.assertNotIn(CONFIG["api_key"], response.text)
        self.assertEqual(CONFIG["api_key"], self.record(headers)["api_key"])

    def test_empty_null_and_omitted_key_preserve_saved_secret(self):
        self.save()
        for replacement in ("", "   ", None):
            with self.subTest(replacement=replacement):
                response = self.save(dict(CONFIG, api_key=replacement, temperature=0.7))
                self.assertEqual(200, response.status_code)
                self.assertEqual(CONFIG["api_key"], self.record()["api_key"])
                self.assertEqual(0.7, self.record()["temperature"])
        update = {key: value for key, value in CONFIG.items() if key != "api_key"}
        self.assertEqual(200, self.save(update).status_code)
        self.assertEqual(CONFIG["api_key"], self.record()["api_key"])

    def test_explicit_new_key_replaces_saved_value(self):
        self.save()
        self.assertEqual(200, self.save(dict(CONFIG, api_key="test-replacement")).status_code)
        self.assertEqual("test-replacement", self.record()["api_key"])

    def test_invalid_config_does_not_deactivate_existing_configuration(self):
        self.save()
        before = self.record()
        for update in ({"model": "   "}, {"base_url": ""},
                       {"base_url": "https://other.test/v1", "api_key": ""}):
            with self.subTest(update=update):
                response = self.save(dict(CONFIG, **update))
                self.assertEqual(400, response.status_code)
                self.assertEqual(before, self.record())
                self.assertNotIn(CONFIG["api_key"], response.text)

    def test_first_remote_configuration_requires_key_but_local_service_does_not(self):
        self.assertEqual(400, self.save(dict(CONFIG, api_key="")).status_code)
        self.assertIsNone(self.record())
        self.assertEqual(200, self.save(dict(CONFIG, base_url="http://127.0.0.1:11434/v1", api_key="")).status_code)
        self.assertEqual("", self.record()["api_key"])

    def test_blank_key_cannot_borrow_another_account_or_anonymous_secret(self):
        self.save()
        database.save_llm_config(dict(CONFIG, api_key="anonymous-secret"))
        other_headers = self.login("配置用户乙")
        self.assertFalse(self.client.get("/api/llm-config", headers=other_headers).json()["configured"])
        self.assertEqual(400, self.save(dict(CONFIG, api_key=""), other_headers).status_code)
        self.assertIsNone(self.record(other_headers))
        self.assertEqual(200, self.save(dict(CONFIG, api_key="other-secret"), other_headers).status_code)
        self.assertEqual(CONFIG["api_key"], self.record()["api_key"])
        self.assertEqual("other-secret", self.record(other_headers)["api_key"])

    def test_invalid_session_does_not_read_or_write_anonymous_config(self):
        database.save_llm_config(dict(CONFIG, api_key="anonymous-secret"))
        before = database.get_active_llm_config()
        invalid = {"X-Session-Token": "expired-test-token"}
        self.assertEqual(401, self.client.get("/api/llm-config", headers=invalid).status_code)
        self.assertEqual(401, self.save(headers=invalid).status_code)
        self.assertEqual(before, database.get_active_llm_config())

    def test_database_path_is_stable_from_another_working_directory(self):
        # 独立进程只导入路径常量，不初始化或改写实际项目数据库。
        environment = dict(os.environ, PYTHONPATH=str(ROOT))
        output = subprocess.check_output(
            [sys.executable, "-B", "-c", "from api.database import DB_PATH; print(DB_PATH)"],
            cwd=self.temp.name, env=environment, text=True,
        )
        self.assertEqual(ROOT / "data/optiagent.sqlite3", Path(output.strip()))

    def test_logged_in_http_and_sse_prefer_controller_even_with_legacy_server_default(self):
        self.save()
        user_id = database.get_user_by_token(self.headers["X-Session-Token"])["id"]
        # 只模拟主控返回，验证两个公开入口确实选用当前账户配置。
        result = {"answer": "由 LLM 驱动", "status": "ANALYSIS",
                  "agent_controller": {"mode": "llm"}}
        with patch.dict(os.environ, {"OPTIAGENT_AGENT_MODE": "legacy"}), patch(
            "api.services.llm_controller.run_llm_controller", return_value=result
        ) as controller:
            for route in ("/api/ask", "/api/ask/stream"):
                with self.subTest(route=route):
                    response = self.client.post(route, headers=self.headers, json={"question": "分析任务"})
                    self.assertEqual(200, response.status_code)
                    if route == "/api/ask":
                        self.assertEqual("llm", response.json()["agent_controller"]["mode"])
                    else:
                        self.assertIn('"mode": "llm"', response.text)
                    self.assertEqual(user_id, controller.call_args.kwargs["user_id"])
                    self.assertEqual(CONFIG["api_key"], controller.call_args.kwargs["llm_config"].api_key)

    def test_explicit_legacy_request_remains_available(self):
        self.save()
        with patch("api.services.llm_controller.run_llm_controller") as controller, patch(
            "api.services.agent_workflow.AGENT_GRAPH.invoke",
            return_value={"result": {"answer": "显式旧模式", "status": "ANALYSIS"}}
        ):
            response = self.client.post("/api/ask", headers=self.headers,
                                        json={"question": "分析任务", "agent_mode": "legacy"})
        self.assertEqual(200, response.status_code)
        self.assertEqual("legacy", response.json()["agent_controller"]["mode"])
        controller.assert_not_called()

    def test_model_directory_uses_saved_account_key_without_changing_config(self):
        self.save()
        before = self.record()
        response = Mock(status_code=200)
        response.json.return_value = {"data": [{"id": "model-b"}, {"id": "model-a"}, {"id": "model-b"}]}
        with patch("optiagent.llm.requests.get", return_value=response) as get:
            result = self.client.post("/api/llm-models", headers=self.headers,
                                      json={"base_url": CONFIG["base_url"], "api_key": None})
        self.assertEqual(200, result.status_code)
        self.assertEqual(["model-a", "model-b"], result.json()["models"])
        self.assertEqual(CONFIG["base_url"] + "/models", get.call_args.args[0])
        self.assertEqual("Bearer " + CONFIG["api_key"], get.call_args.kwargs["headers"]["Authorization"])
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        self.assertNotIn(CONFIG["api_key"], result.text)
        self.assertEqual(before, self.record())

    def test_new_service_requires_its_own_key_and_does_not_save_query(self):
        self.save()
        before = self.record()
        with patch("api.main.list_openai_compatible_models", return_value=["new-model"]) as discover:
            denied = self.client.post("/api/llm-models", headers=self.headers,
                                      json={"base_url": "https://other.test/v1"})
            self.assertEqual(400, denied.status_code)
            discover.assert_not_called()
            result = self.client.post("/api/llm-models", headers=self.headers,
                                      json={"base_url": "https://other.test/v1", "api_key": "new-query-secret"})
        self.assertEqual(200, result.status_code)
        self.assertEqual("new-query-secret", discover.call_args.args[0].api_key)
        self.assertEqual(before, self.record())

    def test_directory_cannot_borrow_another_account_or_expired_session(self):
        self.save()
        other = self.login("模型目录用户乙")
        with patch("api.main.list_openai_compatible_models") as discover:
            result = self.client.post("/api/llm-models", headers=other, json={"base_url": CONFIG["base_url"]})
            self.assertEqual(400, result.status_code)
            for method in (self.client.get, self.client.post):
                options = {"json": {"base_url": CONFIG["base_url"]}} if method == self.client.post else {}
                result = method("/api/llm-models", headers={"X-Session-Token": "expired"}, **options)
                self.assertEqual(401, result.status_code)
            discover.assert_not_called()

    def test_local_directory_can_be_queried_before_saving_model(self):
        with patch("api.main.list_openai_compatible_models", return_value=["local-model"]) as discover:
            result = self.client.post("/api/llm-models", headers=self.headers,
                                      json={"base_url": "http://127.0.0.1:11434/v1"})
        self.assertEqual(200, result.status_code)
        self.assertEqual("", discover.call_args.args[0].api_key)
        self.assertIsNone(self.record())

    def test_directory_errors_do_not_expose_service_body_or_credentials(self):
        self.save()
        response = Mock(status_code=401)
        error = requests.HTTPError("secret-service-body " + CONFIG["api_key"], response=response)
        with patch("api.main.list_openai_compatible_models", side_effect=error):
            result = self.client.get("/api/llm-models", headers=self.headers)
        self.assertEqual(502, result.status_code)
        self.assertIn("认证失败", result.json()["detail"])
        self.assertNotIn(CONFIG["api_key"], result.text)
        self.assertNotIn("secret-service-body", result.text)

    def test_directory_rejects_invalid_urls_without_network_request(self):
        self.save()
        with patch("api.main.list_openai_compatible_models") as discover:
            for address in ("", "file:///etc/passwd", "https://user:password@example.test/v1", "https://example.test/v1?key=secret"):
                with self.subTest(address=address):
                    result = self.client.post("/api/llm-models", headers=self.headers, json={"base_url": address})
                    self.assertEqual(400, result.status_code)
            discover.assert_not_called()


class ModelDirectoryParsingTests(unittest.TestCase):
    """模型目录独立验证兼容响应和 HTTP 边界，不调用外部服务。"""

    def test_full_chat_url_is_normalized_and_model_names_remain_plain_text(self):
        response = Mock(status_code=200)
        response.json.return_value = {"data": [{"id": "<model>"}, {"id": ""}, {"id": 42}, "plain-model"]}
        config = LLMConfig(True, "test-key", "https://example.test/v1/chat/completions/", "")
        with patch("optiagent.llm.requests.get", return_value=response) as get:
            models = list_openai_compatible_models(config)
        self.assertEqual(["<model>", "plain-model"], models)
        self.assertEqual("https://example.test/v1/models", get.call_args.args[0])

    def test_redirect_and_malformed_directory_are_explicitly_rejected(self):
        config = LLMConfig(True, "test-key", "https://example.test/v1", "")
        for status, payload in ((302, {}), (200, {"unexpected": "body"})):
            response = Mock(status_code=status)
            response.json.return_value = payload
            with self.subTest(status=status), patch("optiagent.llm.requests.get", return_value=response):
                with self.assertRaises(ValueError):
                    list_openai_compatible_models(config)


if __name__ == "__main__":
    unittest.main()
