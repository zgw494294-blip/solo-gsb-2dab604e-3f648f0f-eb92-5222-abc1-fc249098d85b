"""线性标定式 y = a·x + b 的加权最小二乘拟合。

POST /fit-calibration 的核心计算：

- 3～16 个标定点各给出 x、x_unit、y、y_unit、y_uncertainty（不确定度与
  y 同单位、恒正），另给出目标单位 x_to、y_to；
- 单位表达式复用 :mod:`app.converter` 的解析器，把观测值与不确定度按
  乘性比例换到目标单位并逐项核对量纲；
- 以 w_i = 1/u_i² 为权，用正规方程

      [Σw   Σwx ] [b]   [Σwy ]
      [Σwx Σwx²] [a] = [Σwxy]

  配合 ``Fraction`` 精确求最小二乘解；系数行列式
  ``Σw·Σwx² − (Σwx)²`` 为零当且仅当换算后的 x 全相同，此时拒绝；
- 残差卡方 ``chi_square = Σ[((y − a·x − b)/u)²]``（无量纲）同样以有理数
  精确累计；slope 单位为 y_to/x_to，intercept 单位为 y_to，按
  ROUND_HALF_EVEN 保留最多 12 位小数的格式化由
  :func:`app.converter.format_result` 完成。

``unit="1"`` 表示无量纲（比例 1）。``°C`` / ``°F`` 是偏移温标，没有唯一
的乘性比例，目标单位与各点单位一律拒绝；x 视为精确量，不带不确定度。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .converter import (
    DimensionMismatchError,
    InvalidValueError,
    OffsetScaleInCompoundError,
    Quantity,
    UnitSyntaxError,
    dim_description,
    parse_expression,
    parse_value,
)


# ---------------------------------------------------------------------------
# 输入结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationPoint:
    """一个标定点：x 精确，y 带与 y 同单位的恒正不确定度。"""
    x: str
    x_unit: str
    y: str
    y_unit: str
    y_uncertainty: str


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------

class CalibrationError(Exception):
    """拟合失败的基类；error 为对外的错误类型标识，extra 为坐标等附加信息。"""

    error = "calibration_error"

    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.message = message
        self.extra = extra


class CalibrationUnitError(CalibrationError):
    """单位表达式无法用于拟合。

    kind 为 "syntax_error" / "offset_temperature_not_allowed"；坐标为
    point（点下标）+ field（x_unit/y_unit），缺省 point 表示目标单位
    （field 为 x_to/y_to）。
    """

    def __init__(self, kind: str, message: str,
                 position: int | None, **coords):
        super().__init__(message, position=position, **coords)
        self.error = kind


class CalibrationNumberError(CalibrationError):
    """数值不是有限十进制字符串，或 y_uncertainty 不严格为正。

    error 为 "invalid_value" / "invalid_uncertainty"；坐标为 point + field。
    """

    def __init__(self, error: str, message: str, point: int, field: str):
        super().__init__(message, point=point, field=field)
        self.error = error


class CalibrationDimensionMismatch(CalibrationError):
    """点的单位量纲与目标单位不一致；坐标为 point + field。"""

    error = "dimension_mismatch"


class ConstantXError(CalibrationError):
    """换算到目标单位后所有 x 全相同，无法拟合斜率。"""

    error = "constant_x_values"


# ---------------------------------------------------------------------------
# 单位与数值解析：复用表达式解析器，"1" 为无量纲，拒绝 °C / °F
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


def _parse_number(raw: str, point: int, field: str) -> Fraction:
    """把有限十进制字符串解析为精确的有理数。"""
    try:
        return parse_value(raw)
    except InvalidValueError as exc:
        raise CalibrationNumberError("invalid_value", str(exc), point, field)


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationResult:
    """换算到目标单位后的拟合结果（均为精确有理数）。

    slope 的单位为 y_to/x_to，intercept 的单位为 y_to；chi_square 无量纲。
    """
    slope: Fraction
    intercept: Fraction
    chi_square: Fraction


def fit_calibration(x_to: str, y_to: str,
                    points: list[CalibrationPoint]) -> CalibrationResult:
    """按 1/u² 加权拟合 y = a·x + b，并计算残差卡方。

    成功返回 CalibrationResult（slope/intercept/chi_square 均为精确有理数）；
    失败抛出 CalibrationError 子类。
    """
    target_x = _parse_unit(x_to, field="x_to")
    target_y = _parse_unit(y_to, field="y_to")

    xs: list[Fraction] = []
    ys: list[Fraction] = []
    us: list[Fraction] = []

    for index, point in enumerate(points):
        x_unit = _parse_unit(point.x_unit, point=index, field="x_unit")
        y_unit = _parse_unit(point.y_unit, point=index, field="y_unit")

        x = _parse_number(point.x, index, "x")
        y = _parse_number(point.y, index, "y")
        try:
            uncertainty = parse_value(point.y_uncertainty)
        except InvalidValueError as exc:
            raise CalibrationNumberError(
                "invalid_uncertainty", str(exc), index, "y_uncertainty")
        if uncertainty <= 0:
            raise CalibrationNumberError(
                "invalid_uncertainty",
                "y_uncertainty must be strictly positive: "
                f"{point.y_uncertainty!r}", index, "y_uncertainty")

        if x_unit.dims != target_x.dims:
            raise CalibrationDimensionMismatch(
                f"dimension mismatch: point {index} x has "
                f"{dim_description(x_unit.dims)} but x_to is "
                f"{dim_description(target_x.dims)}",
                point=index, field="x_unit")
        if y_unit.dims != target_y.dims:
            raise CalibrationDimensionMismatch(
                f"dimension mismatch: point {index} y has "
                f"{dim_description(y_unit.dims)} but y_to is "
                f"{dim_description(target_y.dims)}",
                point=index, field="y_unit")

        # 只允许乘性比例单位（偏移温标已在解析时拒绝）：值与不确定度同比例缩放
        xs.append(x * x_unit.scale / target_x.scale)
        ys.append(y * y_unit.scale / target_y.scale)
        us.append(uncertainty * y_unit.scale / target_y.scale)

    # 正规方程：[Σw  Σwx; Σwx Σwx²]·[b; a] = [Σwy; Σwxy]
    sw = swx = swx2 = swy = swxy = Fraction(0)
    for x, y, u in zip(xs, ys, us):
        w = 1 / (u * u)
        sw += w
        swx += w * x
        swx2 += w * x * x
        swy += w * y
        swxy += w * x * y

    determinant = sw * swx2 - swx * swx
    if determinant == 0:
        raise ConstantXError(
            "all x values are identical after conversion to x_to; "
            "a unique slope cannot be determined")

    intercept = (swy * swx2 - swx * swxy) / determinant
    slope = (sw * swxy - swx * swy) / determinant

    chi_square = Fraction(0)
    for x, y, u in zip(xs, ys, us):
        residual = (y - slope * x - intercept) / u
        chi_square += residual * residual

    return CalibrationResult(slope, intercept, chi_square)
