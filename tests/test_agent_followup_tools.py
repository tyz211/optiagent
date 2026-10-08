"""验证原文修改与方案引用的拒绝边界；模拟模型决策但执行真实本地工具。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from unittest.mock import patch

from api import database
from api.database import (
    compare_and_save_requirement, create_conversation, get_conversation_requirement,
    init_db, save_conversation_requirement, save_run, update_run_result,
)
from api.services.llm_controller import ControllerDecision, run_llm_controller
from api.services.result_evidence import explanation_request, read_result_evidence
from api.services.result_evidence import render_result_explanation
from optiagent.llm import LLMConfig
from optiagent.requirement_analysis import analyze_requirements
from optiagent.requirement_patch import RequirementEdit, RequirementPatch, build_patched_requirement, editable_targets


DATA = {"capacity": 5, "items": [{"item": "A", "value": 8, "weight": 3}, {"item": "B", "value": 5, "weight": 2}]}
QUESTION = "求解背包：" + json.dumps({"knapsack": DATA})
SOLVE = ["analyze_requirements", "build_model", "solve", "verify", "finish"]


def edit(quote, *, field="capacity", entity="", target="", value=3, start=0):
    """生成位置可核对的原文证据。"""
    return RequirementEdit(field=field, entity=entity, target=target, value=value,
                           source_quote=quote, source_start=start, source_end=start + len(quote))


def proposal(*edits, revision=1, template="knapsack", mode="solve"):
    """生成同一版本的原子修改批次。"""
    return RequirementPatch(expected_revision=revision, template_id=template, mode=mode, edits=list(edits))


def action(name, **kwargs):
    """脚本仅替换模型的下一步选择，不替代求解与验算。"""
    return ControllerDecision(**{"action": name, "reason": "根据本轮需求与工具反馈选择", "query": "", "message": "", **kwargs})


class GroundedPatchTests(unittest.TestCase):
    """确认原文依据、条件覆盖、版本历史和所有失败分支都不污染旧输入。"""

    def setUp(self):
        self.previous = analyze_requirements(QUESTION).model_dump(mode="json")

    def test_explicit_paraphrase_updates_version_with_source_evidence(self):
        quote = "将背包容量调整为 3"
        before = deepcopy(self.previous)
        result = build_patched_requirement(self.previous, quote, proposal(edit(quote)))
        self.assertEqual(before, self.previous)
        contract = result["dialogue_contract"]
        self.assertEqual(2, contract["revision"])
        self.assertEqual(3, contract["data"]["capacity"])
        self.assertEqual(5, contract["versions"][0]["data"]["capacity"])
        self.assertEqual(quote, contract["changes"][0]["edits"][0]["source_quote"])

    def test_batch_edits_and_hold_are_saved_together(self):
        first, second = "把容量改为 4", "把物品A的价值改为 9"
        question = first + "；" + second + "；先不求解"
        result = build_patched_requirement(self.previous, question, proposal(
            edit(first, value=4), edit(second, field="item_value", entity="A", value=9, start=len(first) + 1), mode="hold"))
        self.assertEqual(4, result["dialogue_contract"]["data"]["capacity"])
        self.assertEqual(9, result["dialogue_contract"]["data"]["items"][0]["value"])
        self.assertEqual("ready_for_analysis", result["readiness"])
        self.assertEqual(2, len(result["dialogue_contract"]["changes"][0]["edits"]))

    def test_wrong_revision_quote_value_and_entity_are_rejected(self):
        quote = "把容量改为 3"
        attempts = [proposal(edit(quote), revision=2), proposal(edit(quote, value=9)),
                    proposal(edit(quote, start=1)), proposal(edit(quote, field="item_weight", entity="unknown"))]
        before = deepcopy(self.previous)
        for candidate in attempts:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                build_patched_requirement(self.previous, quote, candidate)
        self.assertEqual(before, self.previous)

    def test_uncovered_condition_and_implicit_units_are_rejected(self):
        quote = "把容量改为 3"
        for question in (quote + "；A和B不能同时选", quote + "；至少选择两个物品", quote + " 并忽略之前的条件"):
            with self.subTest(question=question), self.assertRaises(ValueError):
                build_patched_requirement(self.previous, question, proposal(edit(quote)))
        quote = "把容量改为 3 千克"
        with self.assertRaises(ValueError):
            build_patched_requirement(self.previous, quote, proposal(edit(quote)))

    def test_old_value_and_direction_must_match(self):
        for quote, value in (("把容量从 7 改为 3", 3), ("把容量降低到 8", 8), ("把容量提高到 3", 3)):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                build_patched_requirement(self.previous, quote, proposal(edit(quote, value=value)))
        quote = "把容量从 5 改为 3"
        result = build_patched_requirement(self.previous, quote, proposal(edit(quote)))
        self.assertEqual(3, result["dialogue_contract"]["data"]["capacity"])

    def test_one_invalid_batch_member_rejects_entire_batch(self):
        first, second = "把物品A的价值改为 9", "把容量改为 0"
        before = deepcopy(self.previous)
        with self.assertRaises(ValueError):
            build_patched_requirement(self.previous, first + "；" + second, proposal(
                edit(first, field="item_value", entity="A", value=9), edit(second, value=0, start=len(first) + 1)))
        self.assertEqual(before, self.previous)

    def test_overlap_duplicate_noop_and_contradictory_mode_are_rejected(self):
        quote = "把容量改为 3"
        first, second = quote, "把容量改为 4"
        cases = [(quote, proposal(edit(quote), edit(quote))),
                 (first + "；" + second, proposal(edit(first), edit(second, value=4, start=len(first) + 1))),
                 (quote, proposal(edit(quote), mode="hold")),
                 (quote + "；先不求解；并求解", proposal(edit(quote), mode="hold")),
                 ("把容量改为 5", proposal(edit("把容量改为 5", value=5)))]
        for question, candidate in cases:
            with self.subTest(question=question), self.assertRaises(ValueError):
                build_patched_requirement(self.previous, question, candidate)

    def test_unresolved_previous_conditions_cannot_be_silently_erased(self):
        self.previous["dialogue_contract"]["error"] = "还需要确认互斥条件"
        quote = "把容量改为 3"
        with self.assertRaises(ValueError):
            build_patched_requirement(self.previous, quote, proposal(edit(quote)))

    def test_patch_does_not_bypass_history_limit_or_drop_old_versions(self):
        contract = self.previous["dialogue_contract"]
        contract["versions"] = [deepcopy(contract["versions"][0]) for _ in range(100)]
        quote = "把容量改为 3"
        with self.assertRaises(ValueError):
            build_patched_requirement(self.previous, quote, proposal(edit(quote)))
        self.assertEqual(100, len(contract["versions"]))

    def test_transportation_unit_entity_and_route_are_grounded(self):
        data = {"suppliers": [{"name": "S", "supply": 5}], "consumers": [{"name": "T", "demand": 4}],
                "routes": [{"source": "S", "target": "T", "cost": 2}], "forbidden_routes": [],
                "quantity_unit": "吨", "currency": "元"}
        previous = analyze_requirements(json.dumps({"transportation": data})).model_dump(mode="json")
        first, second = "把S的供应量改为 8 吨", "把从S到T的单位运费改为 3 元/吨"
        result = build_patched_requirement(previous, first + "；" + second, proposal(
            edit(first, field="supply", entity="S", value=8),
            edit(second, field="route_cost", entity="S", target="T", value=3, start=len(first) + 1), template="transportation"))
        self.assertEqual(8, result["dialogue_contract"]["data"]["suppliers"][0]["supply"])
        self.assertEqual(3, result["dialogue_contract"]["data"]["routes"][0]["cost"])
        quote = "把S的供应量改为 8000 千克"
        with self.assertRaises(ValueError):
            build_patched_requirement(previous, quote, proposal(edit(quote, field="supply", entity="S", value=8000), template="transportation"))

    def test_field_lookup_is_bounded_and_filtered(self):
        rows = editable_targets(self.previous, "A")["targets"]
        self.assertEqual({"item_value", "item_weight"}, {row["field"] for row in rows})
        for index in range(40):
            self.previous["dialogue_contract"]["data"]["items"].append({"item": str(index), "weight": 1, "value": 1})
        result = editable_targets(self.previous)
        self.assertEqual(20, len(result["targets"]))
        self.assertGreater(result["remaining_count"], 0)


class FollowupControllerTests(unittest.TestCase):
    """真实求解验证修改、解释、作用域和原子提交。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(database, "DB_PATH", Path(self.temp.name) / "followup.sqlite3")
        self.db_patch.start()
        init_db()
        self.conversation = create_conversation("追问测试")["id"]
        self.config = LLMConfig(True, "test-secret", "https://example.test/v1", "scripted-model")

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    def run_script(self, actions, question=QUESTION):
        """只有模型响应使用脚本，其余工具执行真实实现。"""
        outputs = [action(item) if isinstance(item, str) else item for item in actions]
        with patch("api.services.llm_controller.choose_action", side_effect=outputs) as choose:
            result = run_llm_controller(question=question, requested_dataset_id=None, mcp_config="",
                user_id=None, conversation_id=self.conversation, llm_config=self.config)
        return result, choose

    def test_structured_patch_changes_actual_solver_input(self):
        self.run_script(SOLVE)
        quote = "将背包容量调整为 3"
        result, choose = self.run_script(["read_requirement", action("apply_requirement_patch", patch=proposal(edit(quote))),
            "build_model", "solve", "verify", "finish"], quote)
        self.assertEqual(8, result["objective_value"])
        self.assertTrue(result["solution_verification"]["passed"])
        self.assertEqual(2, result["requirement_analysis"]["dialogue_contract"]["revision"])
        context = json.loads(choose.call_args_list[2].args[1][1]["content"])
        self.assertTrue(context["task"]["current_turn_analyzed"])
        self.assertNotIn("apply_requirement_patch", context["available_actions"])

    def test_rejected_patch_has_specific_feedback_and_does_not_save_partial_change(self):
        self.run_script(SOLVE)
        before = get_conversation_requirement(self.conversation)
        quote = "把容量改为 3"
        result, choose = self.run_script([action("apply_requirement_patch", patch=proposal(edit(quote))),
            action("clarify", message="请明确互斥要求。")], quote + "；A和B不能同时选")
        self.assertEqual("NEEDS_CLARIFICATION", result["status"])
        self.assertEqual(before, get_conversation_requirement(self.conversation))
        feedback = json.loads(choose.call_args_list[1].args[1][1]["content"])["tool_observations"][-1]
        self.assertIn("未覆盖条件", feedback["validation_error"])

    def test_patch_can_save_without_solving_when_user_says_hold(self):
        self.run_script(SOLVE)
        quote = "把容量改为 3"
        result, choose = self.run_script([action("apply_requirement_patch", patch=proposal(edit(quote), mode="hold")), "solve", "finish"], quote + "；先不求解")
        self.assertEqual("ANALYSIS", result["status"])
        self.assertIsNone(result["objective_value"])
        context = json.loads(choose.call_args_list[1].args[1][1]["content"])
        self.assertNotIn("solve", context["available_actions"])

    def test_explanation_cites_facts_without_rewriting_source_run_or_requirement(self):
        source, _ = self.run_script(SOLVE)
        before = get_conversation_requirement(self.conversation)
        row_before = database.get_scoped_run(source["run_id"], user_id=None, conversation_id=self.conversation)
        result, choose = self.run_script(["read_verified_result", "solve", action("explain_result", fact_ids=["objective", "metric:used_weight", "decision:0"]), "finish"], "解释上一轮结果")
        self.assertEqual("ANALYSIS", result["status"])
        self.assertIn("13", result["answer"])
        self.assertEqual(source["run_id"], result["result_reference"]["run_id"])
        self.assertEqual(3, len(result["fact_citations"]))
        self.assertIsNone(result["objective_value"])
        self.assertNotIn("solution_verification", result)
        self.assertNotEqual(source["run_id"], result["run_id"])
        self.assertEqual(row_before, database.get_scoped_run(source["run_id"], user_id=None, conversation_id=self.conversation))
        self.assertEqual(before, get_conversation_requirement(self.conversation))
        context = json.loads(choose.call_args_list[1].args[1][1]["content"])
        self.assertNotIn("solve", context["available_actions"])

    def test_unknown_fact_is_rejected_and_model_can_correct_selection(self):
        self.run_script(SOLVE)
        result, choose = self.run_script(["read_verified_result", action("explain_result", fact_ids=["made_up_probability"]),
            action("explain_result", fact_ids=["objective", "verification"]), "finish"], "说明当前方案的验算依据")
        self.assertEqual("ANALYSIS", result["status"])
        feedback = json.loads(choose.call_args_list[2].args[1][1]["content"])["tool_observations"][-1]
        self.assertIn("不存在", feedback["validation_error"])

    def test_reading_result_cannot_ignore_an_extra_condition(self):
        self.run_script(SOLVE)
        result, choose = self.run_script(["read_verified_result", action("clarify", message="请明确容量变更。")], "解释上一轮结果；并把容量改为 3")
        first = json.loads(choose.call_args_list[0].args[1][1]["content"])
        self.assertNotIn("read_verified_result", first["available_actions"])
        self.assertEqual("NEEDS_CLARIFICATION", result["status"])

    def test_latest_result_must_match_current_version_but_explicit_history_is_labeled(self):
        first, _ = self.run_script(SOLVE)
        self.run_script(["analyze_requirements", "finish"], "把容量改成 3；先不求解")
        current = get_conversation_requirement(self.conversation)
        with self.assertRaises(ValueError):
            read_result_evidence(user_id=None, conversation_id=self.conversation, brief=current, run_id=None)
        result, _ = self.run_script([action("read_verified_result", run_id=first["run_id"]), action("explain_result", fact_ids=["objective"]), "finish"], f"解释方案 #{first['run_id']}")
        self.assertFalse(result["result_reference"]["matches_current_requirement"])
        self.assertIn("历史方案", result["answer"])

    def test_cross_conversation_user_and_pending_record_cannot_supply_evidence(self):
        source, _ = self.run_script(SOLVE)
        before = get_conversation_requirement(self.conversation)
        other = create_conversation("别人的对话", user_id=999)["id"]
        for user, conversation in ((None, other), (999, self.conversation)):
            with self.subTest(user=user), self.assertRaises(ValueError):
                read_result_evidence(user_id=user, conversation_id=conversation, brief=before, run_id=source["run_id"])
        pending = dict(source, status="PENDING_VERIFICATION")
        run_id = save_run(None, "候选", "候选", pending, conversation_id=self.conversation)
        with self.assertRaises(ValueError):
            read_result_evidence(user_id=None, conversation_id=self.conversation, brief=before, run_id=run_id)
        unaccepted = dict(source, workflow_verification={"passed": False})
        update_run_result(source["run_id"], unaccepted)
        with self.assertRaises(ValueError):
            read_result_evidence(user_id=None, conversation_id=self.conversation, brief=before, run_id=source["run_id"])

    def test_atomic_compare_checks_entire_state_and_scope_not_only_revision(self):
        before = analyze_requirements(QUESTION).model_dump(mode="json")
        save_conversation_requirement(self.conversation, before)
        stale = deepcopy(before)
        stale["summary"] = "同一版本的另一条修改"
        save_conversation_requirement(self.conversation, stale)
        quote = "把容量改为 3"
        candidate = build_patched_requirement(before, quote, proposal(edit(quote)))
        with self.assertRaises(ValueError):
            compare_and_save_requirement(self.conversation, user_id=None, expected=before, replacement=candidate)
        self.assertEqual(stale, get_conversation_requirement(self.conversation))
        with self.assertRaises(PermissionError):
            compare_and_save_requirement(self.conversation, user_id=999, expected=stale, replacement=candidate)

    def test_two_writers_cannot_overwrite_each_other(self):
        before = analyze_requirements(QUESTION).model_dump(mode="json")
        save_conversation_requirement(self.conversation, before)

        def submit(value):
            # 两个写者都以同一旧状态构造候选，只有一个能提交。
            quote = f"把容量改为 {value}"
            candidate = build_patched_requirement(before, quote, proposal(edit(quote, value=value)))
            try:
                compare_and_save_requirement(self.conversation, user_id=None, expected=before, replacement=candidate)
            except ValueError:
                return False
            return True

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(submit, [3, 4]))
        self.assertEqual(1, sum(outcomes))
        current = get_conversation_requirement(self.conversation)["dialogue_contract"]
        self.assertEqual(2, current["revision"])
        self.assertEqual(2, len(current["versions"]))

    def test_evidence_invalidated_after_read_cannot_be_used_for_explanation(self):
        source, _ = self.run_script(SOLVE)
        outputs = iter([action("read_verified_result"), action("explain_result", fact_ids=["objective"]),
                        action("clarify", message="原方案证据已失效，请重新求解。")])
        calls = 0

        def choose(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                # 模拟读取后源记录被撤销，完成前必须再次检查接受状态。
                update_run_result(source["run_id"], dict(source, workflow_verification={"passed": False}))
            return next(outputs)

        with patch("api.services.llm_controller.choose_action", side_effect=choose):
            result = run_llm_controller(question="解释上一轮方案", requested_dataset_id=None, mcp_config="",
                user_id=None, conversation_id=self.conversation, llm_config=self.config)
        self.assertEqual("NEEDS_CLARIFICATION", result["status"])
        self.assertNotIn("fact_citations", result)

    def test_source_invalidated_after_explanation_is_rejected_at_final_return(self):
        source, _ = self.run_script(SOLVE)
        outputs = iter([action("read_verified_result"), action("explain_result", fact_ids=["objective"]),
                        action("finish"), action("clarify", message="源证据已变化，请重新读取。")])
        calls = 0

        def choose(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                # 解释已经生成，但最终返回前源方案不再被接受。
                update_run_result(source["run_id"], dict(source, workflow_verification={"passed": False}))
            return next(outputs)

        with patch("api.services.llm_controller.choose_action", side_effect=choose):
            result = run_llm_controller(question="解释上一轮方案", requested_dataset_id=None, mcp_config="",
                user_id=None, conversation_id=self.conversation, llm_config=self.config)
        self.assertEqual("NEEDS_CLARIFICATION", result["status"])
        rows = database.list_runs(user_id=None, conversation_id=self.conversation)
        rejected = next(row for row in rows if row["status"] == "UNACCEPTED_ANALYSIS")
        self.assertFalse(json.loads(rejected["result_json"])["workflow_verification"]["passed"])

    def test_compatible_api_cannot_omit_required_nullable_fields(self):
        from api.services.llm_controller import choose_action
        raw = json.dumps({"action": "inspect_data", "reason": "检查", "query": "", "message": ""})
        with patch("api.services.llm_controller.call_openai_compatible_chat", return_value=raw):
            with self.assertRaises(ValueError):
                choose_action(self.config, [], timeout=1, max_output_tokens=700)

    def test_selected_facts_must_cover_explicit_question_focus(self):
        self.run_script(SOLVE)
        evidence = read_result_evidence(user_id=None, conversation_id=self.conversation,
            brief=get_conversation_requirement(self.conversation), run_id=None)
        with self.assertRaises(ValueError):
            render_result_explanation(evidence, ["decision:0"], question="解释方案的目标值")
        with self.assertRaises(ValueError):
            render_result_explanation(evidence, ["objective"], question="解释方案的容量使用")

    def test_explicit_run_reference_cannot_be_silently_changed_by_model(self):
        source, _ = self.run_script(SOLVE)
        result, choose = self.run_script([action("read_verified_result", run_id=source["run_id"] + 1),
            action("clarify", message="请核对方案编号。")], f"解释方案 #{source['run_id']}")
        self.assertEqual("NEEDS_CLARIFICATION", result["status"])
        feedback = json.loads(choose.call_args_list[1].args[1][1]["content"])["tool_observations"][-1]
        self.assertIn("明确指令", feedback["validation_error"])

    def test_every_nested_schema_object_is_closed_with_required_nullable_arguments(self):
        schema = ControllerDecision.model_json_schema()

        def check(value):
            # 检查嵌套修改对象，防止兼容接口拒绝有默认值的严格 Schema。
            if isinstance(value, dict):
                if value.get("type") == "object":
                    self.assertFalse(value["additionalProperties"])
                    self.assertEqual(set(value.get("properties", {})), set(value["required"]))
                self.assertNotIn("default", value)
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        check(schema)
        self.assertEqual((False, None), explanation_request("解释方案；必须多选一个物品"))


if __name__ == "__main__":
    # 单文件执行便于本地重复验证，运行时不调用真实模型服务。
    unittest.main()
