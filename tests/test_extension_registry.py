"""验证新增模板和工具可进入真实执行链，无需修改入口分派代码。"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from pydantic import Field

from api import database
from api.services import agent_tool_registry
from api.services.agent_decision import ControllerDecision, QueryParameters
from api.services.agent_tool_registry import AgentTool, available_actions, execute_agent_tool, register_agent_tool
from api.services.agent_tools import AgentToolContext
from api.services.llm_controller import run_llm_controller
from optiagent import template_extensions
from optiagent.generic_solvers import GenericSolveResult
from optiagent.llm import LLMConfig
from optiagent.mcp_contracts import SolveEnvelope, ValidationReport
from optiagent.mcp_servers.solver_server import solver_list_capabilities
from optiagent.optimization_gateway import build_problem_envelope, solve_problem_envelope
from optiagent.problem_spec import ProblemSpec
from optiagent.solution_verifier import verify_solution
from optiagent.solver_registry import GenericSolverAdapter
from optiagent.template_extensions import TemplateExtension, register_template_extension
from optiagent.templates.capabilities import CAPABILITIES, TemplateCapability
from optiagent.templates.registry import get_template, rank_templates, template_ids


def _bounded_spec(question, data, confidence):
    """测试扩展描述一个单变量有界最大化问题。"""
    return ProblemSpec(
        problem_type="LP", display_name="有界收益", objective="最大化三倍数量",
        sets=[], parameters=["upper_bound"], decision_variables=["quantity"],
        constraints=["0 <= quantity <= upper_bound"], recommended_solver="解析求解",
        solver_reason="单变量目标单调递增", data_requirements=[], output_schema=["quantity"],
        template_id="bounded_reward", confidence=confidence,
    )


def _validate_bound(data):
    """负上限的数据在启动求解前明确拒绝。"""
    valid = isinstance(data.get("upper_bound"), (int, float)) and data["upper_bound"] >= 0
    return ValidationReport(valid=valid, errors=[] if valid else ["上限必须非负"])


def _solve_bound(data, data_source="用户数据", warnings=None, time_limit=None):
    """解析求解只产生候选结果，仍要接受 Gateway 的独立复算。"""
    quantity = data["upper_bound"]
    return GenericSolveResult(
        template_id="bounded_reward", display_name="有界收益", status="OPTIMAL",
        objective_value=3 * quantity, objective_label="收益", solver_name="解析求解", summary="已求得候选",
        decisions=[{"quantity": quantity}], warnings=warnings or [], data_source=data_source,
    )


def _verify_bound(data, solution, state):
    """从原始上限和决策重新计算约束与收益，忽略求解器汇总。"""
    quantity = solution.decisions[0]["quantity"]
    state.check("bounds", 0 <= quantity <= data["upper_bound"], "决策超出数量上限")
    state.recomputed_objective = 3 * quantity


def _extension():
    """一次注册携带完整执行能力，无需修改内置模板列表。"""
    return TemplateExtension(
        template=get_template_definition(),
        capability=TemplateCapability(("upper_bound",), True, ("单变量收益最大化",), ("多变量",)),
        validate_data=_validate_bound, verify_decisions=_verify_bound,
        generic_solver=GenericSolverAdapter("bounded_reward", "有界收益", "解析求解", _solve_bound,
                                            lambda question: (json.loads(question), "测试输入", [])),
    )


def get_template_definition():
    """独立构造新模板元数据，使用公开的模板合同。"""
    from optiagent.templates.registry import OptimizationTemplate
    return OptimizationTemplate("bounded_reward", "有界收益", "LP", ["有界收益"], _bounded_spec)


class TemplateExtensionTests(unittest.TestCase):
    """扩展必须贯通发现、校验、求解和独立验算，且不会污染其他测试。"""

    def setUp(self):
        template_extensions.list_template_extensions()
        self.registry_patch = patch.dict(template_extensions._EXTENSIONS)
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)

    def test_new_template_enters_gateway_and_mcp_with_one_registration(self):
        extension = _extension()
        register_template_extension(extension)
        self.assertIn("bounded_reward", template_ids())
        self.assertIs(extension.template, get_template("bounded_reward"))
        self.assertEqual("bounded_reward", rank_templates("有界收益")[0].template_id)
        self.assertEqual(set(template_ids()), set(CAPABILITIES))
        capability = next(item for item in solver_list_capabilities() if item.template_id == "bounded_reward")
        self.assertEqual(["upper_bound"], capability.input_keys)
        problem = build_problem_envelope("bounded_reward", {"upper_bound": 4})
        solved = solve_problem_envelope(problem)
        self.assertEqual(12, solved.objective_value)
        self.assertTrue(solved.solution_verification.passed)
        corrupted = solved.model_copy(update={"objective_value": 999})
        self.assertFalse(verify_solution(problem, corrupted).passed)
        corrupted = solved.model_copy(update={"decisions": [{"quantity": 5}], "objective_value": 15})
        self.assertFalse(verify_solution(problem, corrupted).passed)

    def test_invalid_data_does_not_invoke_registered_solver(self):
        extension = _extension()
        solve = Mock(wraps=_solve_bound)
        register_template_extension(replace(extension, generic_solver=replace(extension.generic_solver, solve=solve)))
        solved = solve_problem_envelope(build_problem_envelope("bounded_reward", {"upper_bound": -1}))
        self.assertEqual("INVALID_DATA", solved.status)
        solve.assert_not_called()

    def test_custom_envelope_solver_uses_same_verification_boundary(self):
        extension = _extension()

        def solve_envelope(problem, report, time_limit):
            """合同适配器无需在 Gateway 内新增模板分支。"""
            return SolveEnvelope(template_id="bounded_reward", status="OPTIMAL", objective_value=6,
                                 summary="合同候选", decisions=[{"quantity": 2}], validation=report)

        register_template_extension(replace(extension, generic_solver=None,
                                             envelope_solver=solve_envelope, envelope_solver_name="合同适配器"))
        solved = solve_problem_envelope(build_problem_envelope("bounded_reward", {"upper_bound": 2}))
        self.assertTrue(solved.solution_verification.passed)

    def test_duplicate_and_incomplete_extensions_are_not_published(self):
        extension = _extension()
        count = len(template_ids())
        invalid = [
            replace(extension, verify_decisions=None),
            replace(extension, capability=None),
            replace(extension, generic_solver=None),
            replace(extension, generic_solver=replace(extension.generic_solver, template_id="other")),
        ]
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                register_template_extension(candidate)
        self.assertEqual(count, len(template_ids()))
        register_template_extension(extension)
        with self.assertRaisesRegex(ValueError, "已注册"):
            register_template_extension(extension)

    def test_fresh_process_import_order_loads_all_builtins(self):
        # 子进程覆盖首次导入路径，已有内置缓存不能掩盖循环依赖或初始化遗漏。
        orders = [
            ["optiagent.generic_solvers", "optiagent.template_extensions"],
            ["optiagent.solution_verifier", "optiagent.optimization_gateway"],
            ["optiagent.templates.registry", "optiagent.mcp_servers.solver_server"],
        ]
        for order in orders:
            script = (
                "import importlib\n"
                f"for name in {order!r}: importlib.import_module(name)\n"
                "from optiagent.templates.registry import template_ids\n"
                "from optiagent.solver_registry import list_generic_solvers\n"
                "assert len(template_ids()) == 8\n"
                "assert len(list_generic_solvers()) == 7\n"
            )
            with self.subTest(order=order):
                subprocess.run([sys.executable, "-B", "-c", script], check=True, capture_output=True, timeout=30)

    def test_registration_before_first_lookup_still_loads_builtins(self):
        # 新进程先注册外部模板，确认内置加载不再用“注册表非空”判断是否完成。
        script = (
            "import sys\n"
            "sys.path.insert(0, 'tests')\n"
            "from test_extension_registry import _extension\n"
            "from optiagent.template_extensions import register_template_extension\n"
            "register_template_extension(_extension())\n"
            "from optiagent.templates.registry import template_ids\n"
            "from optiagent.solver_registry import list_generic_solvers\n"
            "assert len(template_ids()) == 9\n"
            "assert len(list_generic_solvers()) == 8\n"
        )
        subprocess.run([sys.executable, "-B", "-c", script], check=True, capture_output=True, timeout=30)


class RequiredQuery(QueryParameters):
    """测试工具可以加强共享字段约束，在处理函数之前拒绝非法参数。"""

    query: str = Field(min_length=1, max_length=500)


class AgentToolExtensionTests(unittest.TestCase):
    """动态工具通过主控执行、前置检查与审计，终止门控保持受保护。"""

    def setUp(self):
        agent_tool_registry.list_agent_tools()
        self.registry_patch = patch.dict(agent_tool_registry._TOOLS)
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)

    def test_new_tool_is_visible_to_schema_and_runs_in_controller_with_audit(self):
        def inspect_bound(context, decision):
            """扩展观察进入下一次主控反馈，不自行接受求解候选。"""
            context.state["custom_observation"] = decision.query
            return {"summary": "扩展检查完成", "checked_query": decision.query}

        register_agent_tool(AgentTool("inspect_bound", "检查上限", "按 query 检查数量上限",
                                     RequiredQuery, lambda state, counts: not state.get("requirements_done"), inspect_bound))
        self.assertIn("inspect_bound", ControllerDecision.model_json_schema()["properties"]["action"]["enum"])
        with tempfile.TemporaryDirectory() as directory, patch.object(database, "DB_PATH", Path(directory) / "extensions.sqlite3"):
            database.init_db()
            conversation_id = database.create_conversation("扩展测试")["id"]
            question = "求解背包：" + json.dumps({"capacity": 5, "items": [{"item": "A", "value": 8, "weight": 3}]})
            actions = ["inspect_bound", "analyze_requirements", "build_model", "solve", "verify", "finish"]
            decisions = [ControllerDecision(action=name, reason="扩展回归", query="上限" if name == "inspect_bound" else "", message="")
                         for name in actions]
            with patch("api.services.llm_controller.choose_action", side_effect=decisions) as choose:
                result = run_llm_controller(question=question, requested_dataset_id=None, mcp_config="",
                    user_id=None, conversation_id=conversation_id,
                    llm_config=LLMConfig(True, "test-key", "https://example.test/v1", "scripted"))
            self.assertTrue(result["workflow_verification"]["passed"])
            feedback = json.loads(choose.call_args_list[1].args[1][1]["content"])
            self.assertEqual("上限", feedback["tool_observations"][0]["checked_query"])
            self.assertIn("inspect_bound", [node["node_id"] for node in result["agent_graph"]["nodes"]])
            episode = database.get_agent_episode(result["agent_episode_id"], user_id=None)
            self.assertIn("inspect_bound", [step["node_id"] for step in episode["steps"]])
            self.assertNotIn("inspect_bound", available_actions({"requirements_done": True}, Counter()))

    def test_direct_dispatch_rechecks_state_and_parameters_before_handler(self):
        handler = Mock(return_value={"summary": "不应执行"})
        register_agent_tool(AgentTool("restricted_query", "受限检索", "只允许有数据的检索", RequiredQuery,
                                     lambda state, counts: bool(state.get("data_ready")), handler))
        context = AgentToolContext({}, "测试", None, None, None, 1, [])
        decision = ControllerDecision(action="restricted_query", reason="检查", query="", message="")
        with self.assertRaisesRegex(ValueError, "前置条件"):
            execute_agent_tool(context, decision, Counter())
        context.state["data_ready"] = True
        with self.assertRaises(ValueError):
            execute_agent_tool(context, decision, Counter())
        handler.assert_not_called()

    def test_unknown_duplicate_and_reserved_terminal_tools_are_rejected(self):
        with self.assertRaises(ValueError):
            ControllerDecision(action="missing_tool", reason="拒绝", query="", message="")
        existing = agent_tool_registry.get_agent_tool("finish")
        with self.assertRaisesRegex(ValueError, "已注册"):
            register_agent_tool(existing)
        with self.assertRaisesRegex(ValueError, "终止行动"):
            register_agent_tool(replace(existing, name="bypass_verification"))
