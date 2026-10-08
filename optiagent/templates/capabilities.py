"""可执行模板的能力边界，供 MCP、评测和文档共用。"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateCapability:
    """exact 表示具备精确求解路径，实际最优性仍以求解状态为准。"""

    input_keys: tuple[str, ...]
    exact: bool
    supported: tuple[str, ...]
    unsupported: tuple[str, ...]


# 此处描述已经实现的约束，不把知识库中的建模建议计入支持范围。
BUILTIN_CAPABILITIES = {
    "facility_location": TemplateCapability(
        ("warehouses", "customers", "costs"), True,
        ("需求满足", "容量与启用联动", "固定成本", "强制启用/关闭", "最低运营比例"),
        ("多周期库存", "车辆时间窗")),
    "knapsack": TemplateCapability(
        ("items", "capacity"), True, ("单容量", "0-1 选择", "价值最大化"),
        ("数量下限", "互斥", "多维容量")),
    "assignment": TemplateCapability(
        ("resources", "tasks", "costs"), True, ("每任务恰好一个资源", "每资源至多一个任务", "成本最小化"),
        ("技能资格", "班次覆盖人数", "一人多任务")),
    "tsp": TemplateCapability(
        ("distances 或 distance_matrix",), True, ("访问所有节点并返回起点", "距离最小化"),
        ("多车辆", "车辆容量", "时间窗")),
    "job_shop_scheduling": TemplateCapability(
        ("tasks",), True, ("机器互斥", "工序顺序", "最大完工时间最小化"),
        ("可选机器", "交期惩罚", "换型时间")),
    "production_mix": TemplateCapability(
        ("products", "capacities"), True, ("资源容量", "min_qty/max_qty", "全体产量连续或整数", "利润最大化"),
        ("投产固定成本", "多周期库存", "混合变量类型")),
    "linear_program": TemplateCapability(
        ("variables", "objective", "constraints"), True, ("显式线性约束", "连续/整数/二元变量", "有限整数集合"),
        ("非线性表达式", "任意业务语言自动编译")),
    "transportation": TemplateCapability(
        ("suppliers", "consumers", "routes", "quantity_unit", "currency"), True,
        ("供给上限", "需求恰好满足", "显式禁运", "非负连续运输量", "线性运费最小化"),
        ("车辆与时间窗", "线路容量", "整数运输量", "混合单位自动换算", "多商品", "缺货")),
}


class _CapabilityView(Mapping):
    """从完整扩展读取能力描述，保证发现结果与实际执行能力一致。"""

    def __getitem__(self, template_id: str) -> TemplateCapability:
        from optiagent.template_extensions import get_template_extension
        extension = get_template_extension(template_id)
        if extension is None:
            raise KeyError(template_id)
        return extension.capability

    def __iter__(self):
        from optiagent.template_extensions import list_template_extensions
        return iter(extension.template.template_id for extension in list_template_extensions())

    def __len__(self):
        from optiagent.template_extensions import list_template_extensions
        return len(list_template_extensions())


CAPABILITIES = _CapabilityView()
