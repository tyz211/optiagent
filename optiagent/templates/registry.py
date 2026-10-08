"""模板查询的兼容入口，所有查询使用统一扩展注册表。"""

from __future__ import annotations

from collections.abc import Sequence

from optiagent.data import SupplyChainData
from optiagent.templates.definitions import OptimizationTemplate, SpecBuilder


def list_templates() -> list[OptimizationTemplate]:
    """返回已完整注册的模板，包括应用启动时加入的扩展。"""
    from optiagent.template_extensions import list_template_extensions
    return [extension.template for extension in list_template_extensions()]


def get_template(template_id: str) -> OptimizationTemplate:
    """按标识取得模板，未注册模板保持原有异常合同。"""
    from optiagent.template_extensions import get_template_extension
    extension = get_template_extension(template_id)
    if extension is None:
        raise KeyError(f"Unknown optimization template: {template_id}")
    return extension.template


def rank_templates(question: str, data: SupplyChainData | None = None) -> list[OptimizationTemplate]:
    """沿用原有打分规则，同时纳入新注册模板。"""
    return sorted(list_templates(), key=lambda item: item.score(question, data), reverse=True)


def template_ids() -> list[str]:
    """所有路由与需求提示词使用同一份模板标识。"""
    return [template.template_id for template in list_templates()]


class _TemplateView(Sequence):
    """兼容旧的 TEMPLATES 只读序列，避免维护第二份运行时列表。"""

    def __getitem__(self, index):
        return list_templates()[index]

    def __len__(self):
        return len(list_templates())


TEMPLATES = _TemplateView()
