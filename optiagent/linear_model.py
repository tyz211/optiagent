from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# 数学文本先转成受限数据合同，绝不把用户表达式交给 eval 或代码执行。
TEMPLATE_ID = "linear_program"
MAX_TEXT = 30_000
NAME = r"[A-Za-z][A-Za-z0-9_]*"
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TOKEN = re.compile(rf"\s*({NUMBER}|{NAME}|[()+*/-])")


class ContractModel(BaseModel):
    """拒绝未知字段和非有限数值，避免模型含义被静默改变。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LinearVariable(ContractModel):
    """变量类型、边界和可选离散取值集合。"""

    name: str = Field(pattern=rf"^{NAME}$", max_length=64)
    kind: Literal["binary", "integer", "continuous"]
    lower: float | None = None
    upper: float | None = None
    values: list[float] | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def check_domain(self):
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"{self.name} 的下界大于上界")
        if self.values is not None:
            if self.kind == "continuous" or any(not v.is_integer() for v in self.values):
                raise ValueError("有限取值集合目前只支持整数；请明确变量类型")
            if len(set(self.values)) != len(self.values):
                raise ValueError(f"{self.name} 的取值集合存在重复")
            if any((self.lower is not None and v < self.lower) or
                   (self.upper is not None and v > self.upper) for v in self.values):
                raise ValueError(f"{self.name} 的取值集合与边界冲突")
        if self.kind == "binary" and (
            (self.lower is not None and self.lower not in (0, 1)) or
            (self.upper is not None and self.upper not in (0, 1)) or
            (self.values is not None and not set(self.values) <= {0, 1})
        ):
            raise ValueError("二元变量只能取 0 或 1")
        return self


class LinearExpression(ContractModel):
    """仿射表达式保留常数项，避免目标常数或右端变量丢失。"""

    coefficients: dict[str, float]
    constant: float = 0.0


class LinearObjective(LinearExpression):
    sense: Literal["max", "min"]


class LinearConstraint(LinearExpression):
    sense: Literal["<=", ">=", "="]
    rhs: float
    label: str = Field(default="", max_length=1000)


class LinearModel(ContractModel):
    """可由文本或 JSON 提交的 LP/MILP 合同。"""

    variables: list[LinearVariable] = Field(min_length=1, max_length=200)
    objective: LinearObjective
    constraints: list[LinearConstraint] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def check_references(self):
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("变量声明重复")
        for expression in [self.objective, *self.constraints]:
            unknown = set(expression.coefficients) - set(names)
            if unknown:
                raise ValueError("请声明以下变量的取值范围：" + "、".join(sorted(unknown)))
        return self


@dataclass
class Affine:
    """解析阶段仅保存系数与常数，后续由合同统一检查引用。"""

    coefficients: dict[str, float]
    constant: float = 0.0

    def plus(self, other: Affine, sign: float = 1) -> Affine:
        result = dict(self.coefficients)
        for name, value in other.coefficients.items():
            result[name] = result.get(name, 0.0) + sign * value
        return Affine(result, self.constant + sign * other.constant)

    def scale(self, factor: float) -> Affine:
        return Affine({name: value * factor for name, value in self.coefficients.items()}, self.constant * factor)

    @property
    def scalar(self) -> bool:
        return not any(self.coefficients.values())


def parse_expression(text: str) -> Affine:
    """递归下降解析线性表达式；乘法只能有一个非常数因子。"""

    tokens = []
    offset = 0
    text = text.strip()
    while offset < len(text):
        match = TOKEN.match(text, offset)
        if not match:
            raise ValueError(f"不支持的表达式片段：{text[offset:offset + 40]}；请使用线性加减和常数乘除")
        tokens.append(match[1])
        offset = match.end()
    position = 0

    def atom(depth: int) -> Affine:
        nonlocal position
        if depth > 40 or position >= len(tokens):
            raise ValueError("表达式不完整或括号嵌套过深")
        token = tokens[position]
        position += 1
        if token in {"+", "-"}:
            return atom(depth + 1).scale(-1 if token == "-" else 1)
        if token == "(":
            value = addition(depth + 1)
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("表达式括号不匹配")
            position += 1
            return value
        if re.fullmatch(NUMBER, token):
            value = float(token)
            if not math.isfinite(value):
                raise ValueError("表达式包含非有限数值")
            return Affine({}, value)
        if re.fullmatch(NAME, token):
            return Affine({token: 1.0})
        raise ValueError(f"不完整的表达式：{token}")

    def product(depth: int) -> Affine:
        nonlocal position
        value = atom(depth)
        while position < len(tokens) and tokens[position] not in {"+", "-", ")"}:
            token = tokens[position]
            explicit = token in {"*", "/"}
            if not explicit and re.fullmatch(NUMBER, token):
                raise ValueError("相邻常数之间缺少运算符")
            if explicit:
                position += 1
            other = atom(depth)
            if token == "/":
                if not other.scalar or other.constant == 0:
                    raise ValueError("线性表达式只能除以非零常数")
                value = value.scale(1 / other.constant)
            elif value.scalar:
                value = other.scale(value.constant).plus(Affine({k: 0 for k in value.coefficients}))
            elif other.scalar:
                value = value.scale(other.constant).plus(Affine({k: 0 for k in other.coefficients}))
            else:
                raise ValueError("检测到变量相乘；当前入口仅支持线性规划，不会忽略非线性项")
        return value

    def addition(depth: int) -> Affine:
        nonlocal position
        value = product(depth)
        while position < len(tokens) and tokens[position] in {"+", "-"}:
            sign = -1 if tokens[position] == "-" else 1
            position += 1
            value = value.plus(product(depth), sign)
        return value

    result = addition(0)
    if position != len(tokens):
        raise ValueError("表达式包含多余符号或未匹配括号")
    if not all(math.isfinite(v) for v in [result.constant, *result.coefficients.values()]):
        raise ValueError("表达式计算结果超出有限数值范围")
    return result


def looks_like_linear_model(text: str) -> bool:
    """只识别明确的数学建模信号，避免干扰原有业务模板对话。"""

    return bool(re.search(r"(?:\\(?:max|min)\b|\b(?:max|min|maximize|minimize)\b|目标函数|(?:最大化|最小化)\s*[A-Za-z0-9(]|[A-Za-z]_?\{?\d*\}?\s*(?:\\in|∈|\bin\b))", text))


def _normalize(text: str) -> str:
    """统一常见 LaTeX/Unicode 写法，保留无法解释的内容供报错。"""

    text = re.sub(r"```(?:latex|math|text)?", "\n", text)
    text = re.sub(r"\\(?:begin|end)\{(?:aligned|align\*?|equation\*?|gathered)\}", "\n", text)
    for delimiter in (r"\[", r"\]", r"\(", r"\)", "$$", "$"):
        text = text.replace(delimiter, "\n")
    text = text.replace(r"\\", "\n")
    text = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", text)
    text = re.sub(r"\\(?:quad|qquad|displaystyle)(?![A-Za-z])", " ", text)
    text = re.sub(r"\\(?:text|mathrm)\{(max|min)\}", r"\1", text)
    text = re.sub(r"\\mathbb\{([RZ])\}", lambda m: "reals" if m[1] == "R" else "integers", text)
    text = re.sub(rf"\\(?:dfrac|tfrac|frac)\{{([+-]?{NUMBER})\}}\{{([+-]?{NUMBER})\}}", r"(\1/\2)", text)
    text = re.sub(r"([A-Za-z])_\{(\d+)\}", r"\1\2", text)
    text = re.sub(r"([A-Za-z])_(\d+)", r"\1\2", text)
    for pattern, replacement in ((r"\\leq?(?![A-Za-z])", "<="), (r"\\geq?(?![A-Za-z])", ">="),
                                 (r"\\in(?![A-Za-z])", " in "), (r"\\(?:cdot|times)(?![A-Za-z])", "*"),
                                 (r"\\(max|min)(?![A-Za-z])", r"\1 ")):
        text = re.sub(pattern, replacement, text)
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    text = text.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    return text.translate(str.maketrans({"≤": "<=", "≥": ">=", "∈": " in ", "−": "-", "×": "*",
                                        "，": ",", "；": "\n", ";": "\n", "：": ":", "&": ""}))


def parse_linear_text(text: str) -> dict | None:
    """完整解析数学模型；未识别返回 None，任何未理解的模型行都会报错。"""

    if not looks_like_linear_model(text):
        return None
    if len(text) > MAX_TEXT:
        raise ValueError("数学模型文本过长，请限制在 30000 字符以内")
    variables = []
    constraints = []
    objective = None
    pending_sense = None
    normalized = _normalize(text)
    for raw in normalized.splitlines():
        line = raw.strip().strip("。:").strip()
        # 仅剥离明确的标题和引导词，不吞掉附带的未知业务约束。
        line = re.sub(r"^(?:以及逻辑约束|逻辑约束|目标函数|决策变量|变量范围|变量|满足|约束条件|约束|以及|设|s\.t\.|subject to)\s*:?\s*", "", line, flags=re.I).strip()
        if not line or re.fullmatch(r"(?:请)?(?:求解|求解以下模型|求解以下问题|先不求解|暂不求解|只分析|不要求解)[。:]?", line):
            continue
        domain = re.fullmatch(rf"({NAME}(?:\s*,\s*{NAME})*)\s+in\s+(\{{[^{{}}]+\}}|integers|reals|Z|R)", line)
        if domain:
            names = [name.strip() for name in domain[1].split(",")]
            declared = domain[2]
            if declared.startswith("{"):
                values = []
                for item in declared[1:-1].split(","):
                    item = item.strip()
                    if not re.fullmatch(rf"[+-]?{NUMBER}", item):
                        raise ValueError("请显式列出有限整数集合的每个值，例如 {0,1,2}；暂不支持省略号")
                    values.append(float(item))
                kind = "binary" if set(values) == {0, 1} else "integer"
                variables.extend(dict(name=name, kind=kind, values=values, lower=min(values), upper=max(values)) for name in names)
            else:
                variables.extend(dict(name=name, kind="integer" if declared in {"integers", "Z"} else "continuous") for name in names)
            continue
        goal = re.match(r"^(maximize|minimize|max|min|最大化|最小化)\s*:?\s*(.*)$", line, re.I)
        if goal:
            if objective is not None or pending_sense is not None:
                raise ValueError("检测到多个目标函数，请明确单一优化目标")
            pending_sense = "max" if goal[1].lower() in {"max", "maximize", "最大化"} else "min"
            line = goal[2].strip()
            if not line:
                continue
        if pending_sense is not None:
            line = re.sub(r"^(?:z|f)\s*=\s*", "", line, flags=re.I)
            expression = parse_expression(line)
            objective = dict(sense=pending_sense, coefficients=expression.coefficients, constant=expression.constant)
            pending_sense = None
            continue
        parts = re.split(r"(<=|>=|=)", line)
        if len(parts) in {3, 5}:
            # 链式上下界拆成两条约束，右端变量移项且保留常数。
            for index in range(0, len(parts) - 2, 2):
                left = parse_expression(parts[index])
                right = parse_expression(parts[index + 2])
                expression = left.plus(right, -1)
                constraints.append(dict(coefficients=expression.coefficients, constant=0,
                                        sense=parts[index + 1], rhs=0.0 if expression.constant == 0 else -expression.constant, label=line))
            continue
        raise ValueError(f"无法完整理解模型中的这一行：{line[:160]}。请使用明确的变量声明、目标函数或线性等式/不等式")
    if objective is None or pending_sense is not None:
        raise ValueError("缺少完整目标函数，请提供 max/min 及其线性表达式")
    if not variables:
        raise ValueError("请声明变量取值，例如 x1,x2 in {0,1}，或 x1 in R / x1 in Z")
    return LinearModel.model_validate(dict(variables=variables, objective=objective, constraints=constraints)).model_dump(mode="json")
