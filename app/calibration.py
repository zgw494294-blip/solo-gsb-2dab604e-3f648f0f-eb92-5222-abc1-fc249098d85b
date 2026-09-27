"""线性标定拟合 y = a·x + b。

POST /fit-calibration 的核心计算：

- 3～16 个观测点各给出 x / x_unit / y / y_unit / y_uncertainty；
  x 视为精确，y_uncertainty > 0 且与 y 同单位。
- 复用单位表达式解析器，把各点的 x、y 与不确定度按乘性比例换算到
  目标单位 x_to / y_to，并逐点核对量纲；``unit="1"`` 表示无量纲。
- 按 1/u² 加权，用 ``Fraction`` 精确求最小二乘解：
  正规方程 [[Σw, Σwx], [Σwx, Σwx²]]·[b, a]ᵀ = [Σwy, Σwxy]；
  换算后的 x 不全相同（Σw·Σwx² − (Σwx)² > 0），否则拒绝。
- 同时精确累计 χ² = Σ[(y − a·x − b)/u]²。

``°C`` / ``°F`` 是偏移温标，没有唯一的乘性比例，目标单位与各点单位
一律拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .converter import (
    InvalidValueError,
    OffsetScaleInCompoundError,
    Quantity,
    UnitSyntaxError,
    dim_description,
    parse_expression,
    parse_value,
)


# ---------------------------------------------------------------------------
# 输入与结果
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationPointInput:
    """一个观测点：x 精确，y_uncertainty 与 y 同单位且为正。"""
    x: str
    x_unit: str
    y: str
    y_unit: str
    y_uncertainty: str


@dataclass(frozen=True)
class CalibrationResult:
    """目标单位下的标定结果（均为精确有理数）：斜率、截距与 χ²。"""
    slope: Fraction
    intercept: Fraction
    chi_square: Fraction


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------

class FitError(Exception):
    """拟合失败的基类；error 为对外的错误类型标识，extra 为坐标等附加信息。"""

    error = "fit_error"

    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.message = message
        self.extra = extra


class CalibrationUnitError(FitError):
    """单位表达式无法用于拟合。

    kind 为 "syntax_error" / "offset_temperature_not_allowed"；坐标为
    point（点下标），缺省表示目标单位 x_to / y_to（由 field 指明）。
    """

    def __init__(self, kind: str, message: str,
                 position: int | None, **coords):
        super().__init__(message, position=position, **coords)
        self.error = kind


class InvalidNumberError(FitError):
    """x / y / y_uncertainty 不是有限十进制字符串；坐标为 point。"""
    error = "invalid_value"


class InvalidUncertaintyError(FitError):
    """y_uncertainty 非法（不是有限十进制字符串或不大于 0）；坐标为 point。"""
    error = "invalid_uncertainty"


class PointDimensionMismatchError(FitError):
    """点的单位量纲与目标单位不符；坐标为 point。"""
    error = "dimension_mismatch"


class AllXEqualError(FitError):
    """换算到目标单位后所有 x 相同，无法确定斜率。"""
    error = "all_x_equal"


# ---------------------------------------------------------------------------
# 单位解析：复用表达式解析器，"1" 为无量纲，拒绝 °C / °F
# ---------------------------------------------------------------------------

def _parse_unit(expr: str, **coords) -> Quantity:
    """解析单位表达式；"1" 表示无量纲，偏移温标一律拒绝。"""
    if expr.strip() == "1":
        return Quantity((0, 0, 0, 0), Fraction(1))
    try:
        quantity = parse_expression(expr)
    except UnitSyntaxError as exc:
        raise CalibrationUnitError(
            "syntax_error", exc.message, exc.position, **coords)
    except OffsetScaleInCompoundError as exc:
        raise CalibrationUnitError(
            "offset_temperature_not_allowed", str(exc), exc.position, **coords)
    if quantity.offset_unit is not None:
        raise CalibrationUnitError(
            "offset_temperature_not_allowed",
            f"offset temperature scale {quantity.offset_unit!r} is not "
            "allowed; use K", quantity.pos, **coords)
    return quantity


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def fit_calibration(points: list[CalibrationPointInput],
                    x_to: str, y_to: str) -> CalibrationResult:
    """按 1/u² 加权精确拟合 y = a·x + b。

    成功返回目标单位下的斜率（y_to/x_to）、截距（y_to）与 χ²；
    失败抛出 FitError 子类。
    """
    target_x = _parse_unit(x_to, field="x_to")
    target_y = _parse_unit(y_to, field="y_to")

    xs: list[Fraction] = []
    ys: list[Fraction] = []
    us: list[Fraction] = []

    for index, point in enumerate(points):
        coords = {"point": index}
        try:
            x = parse_value(point.x)
        except InvalidValueError as exc:
            raise InvalidNumberError(str(exc), **coords)
        try:
            y = parse_value(point.y)
        except InvalidValueError as exc:
            raise InvalidNumberError(str(exc), **coords)
        try:
            uncertainty = parse_value(point.y_uncertainty)
        except InvalidValueError as exc:
            raise InvalidUncertaintyError(str(exc), **coords)
        if uncertainty <= 0:
            raise InvalidUncertaintyError(
                "y_uncertainty must be greater than 0", **coords)

        x_unit = _parse_unit(point.x_unit, **coords)
        y_unit = _parse_unit(point.y_unit, **coords)
        if x_unit.dims != target_x.dims:
            raise PointDimensionMismatchError(
                f"dimension mismatch: x unit is "
                f"{dim_description(x_unit.dims)} but x_to is "
                f"{dim_description(target_x.dims)}", **coords)
        if y_unit.dims != target_y.dims:
            raise PointDimensionMismatchError(
                f"dimension mismatch: y unit is "
                f"{dim_description(y_unit.dims)} but y_to is "
                f"{dim_description(target_y.dims)}", **coords)

        # 乘性比例换算到目标单位（偏移温标已在解析阶段拒绝）
        xs.append(x * x_unit.scale / target_x.scale)
        ys.append(y * y_unit.scale / target_y.scale)
        us.append(uncertainty * y_unit.scale / target_y.scale)

    # 加权最小二乘（权重 w = 1/u²）：
    # [Σw   Σwx ] [b]   [Σwy ]
    # [Σwx  Σwx²] [a] = [Σwxy]
    sw = swx = swy = swxx = swxy = Fraction(0)
    for x, y, u in zip(xs, ys, us):
        w = 1 / (u * u)
        sw += w
        swx += w * x
        swy += w * y
        swxx += w * x * x
        swxy += w * x * y

    determinant = sw * swxx - swx * swx
    if determinant == 0:
        raise AllXEqualError(
            "all x values are identical after unit conversion; "
            "the slope cannot be determined")

    slope = (sw * swxy - swx * swy) / determinant
    intercept = (swxx * swy - swx * swxy) / determinant

    chi_square = Fraction(0)
    for x, y, u in zip(xs, ys, us):
        residual = (y - slope * x - intercept) / u
        chi_square += residual * residual

    return CalibrationResult(slope=slope, intercept=intercept,
                             chi_square=chi_square)
