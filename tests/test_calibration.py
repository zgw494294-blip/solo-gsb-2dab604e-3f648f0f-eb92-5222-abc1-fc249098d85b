"""线性标定拟合核心逻辑测试（不经过 HTTP）。"""

from fractions import Fraction

import pytest

from app.calibration import (
    AllXEqualError,
    CalibrationPointInput,
    InvalidNumberError,
    InvalidUncertaintyError,
    PointDimensionMismatchError,
    CalibrationUnitError,
    fit_calibration,
)


def P(x="0", x_unit="m", y="0", y_unit="m", y_uncertainty="1"):
    return CalibrationPointInput(
        x=x, x_unit=x_unit, y=y, y_unit=y_unit,
        y_uncertainty=y_uncertainty)


def fit(points, x_to="m", y_to="m"):
    return fit_calibration(points, x_to, y_to)


# ---------------------------------------------------------------- 成功拟合

def test_exact_line():
    # y = 2x + 1，残差为零 → χ² = 0
    result = fit([P("0", y="1"), P("1", y="3"), P("2", y="5")])
    assert result.slope == Fraction(2)
    assert result.intercept == Fraction(1)
    assert result.chi_square == Fraction(0)


def test_weighted_least_squares_unweighted():
    # y = 2x + 1 + 噪声，等权
    result = fit([
        P("0", y="1.1"), P("1", y="2.9"), P("2", y="5.2"),
    ])
    # 闭式最小二乘：a = 2.05, b = 61/60
    assert result.slope == Fraction(41, 20)
    assert result.intercept == Fraction(61, 60)
    # 残差 (1/12, -1/6, 1/12) → χ² = 1/24
    assert result.chi_square == Fraction(1, 24)


def test_weights_favor_small_uncertainty():
    # 同样的 y 值，不确定度小的点权重大
    equal = fit([P("0", y="1"), P("1", y="3"), P("2", y="4")])
    weighted = fit([
        P("0", y="1", y_uncertainty="1"),
        P("1", y="3", y_uncertainty="1"),
        P("2", y="4", y_uncertainty="1000"),
    ])
    # 前两点精确落在 y = 2x + 1；第三个点权重仅 10⁻⁶，结果几乎不变
    assert weighted.slope != equal.slope
    assert abs(weighted.slope - 2) < Fraction(1, 100_000)
    assert abs(weighted.intercept - 1) < Fraction(1, 100_000)
    assert 0 < weighted.chi_square < Fraction(2, 1_000_000)


def test_negative_x_and_y():
    result = fit([
        P("-1", y="-3"), P("0", y="-1"), P("1", y="1"),
    ])
    assert result.slope == Fraction(2)
    assert result.intercept == Fraction(-1)


def test_x_and_uncertainty_converted_to_target_units():
    # x 以 cm 给出（1cm、2cm、3cm），目标 m；y 以 mm 给出（10mm、30mm、50mm）
    # 不确定度 10mm；换算后即 (0.01,0.02,0.03) m 与 (0.01,0.03,0.05) m
    result = fit([
        P("1", "cm", "10", "mm", "10"),
        P("2", "cm", "30", "mm", "10"),
        P("3", "cm", "50", "mm", "10"),
    ], x_to="m", y_to="m")
    # a = 2（无量纲），b = -0.01 m
    assert result.slope == Fraction(2)
    assert result.intercept == Fraction(-1, 100)
    assert result.chi_square == Fraction(0)


def test_slope_units_follow_from_targets():
    # x: s, y: m；换算只改数值，斜率单位语义上为 m/s
    result = fit([
        P("0", "s", "0", "m", "1"),
        P("1", "s", "10", "m", "1"),
        P("2", "s", "20", "m", "1"),
    ], x_to="s", y_to="m")
    assert result.slope == Fraction(10)
    assert result.intercept == Fraction(0)


def test_dimensionless_one():
    result = fit([
        P("0", "1", "1", "1", "0.5"),
        P("1", "1", "3", "1", "0.5"),
        P("2", "1", "5", "1", "0.5"),
    ], x_to="1", y_to="1")
    assert result.slope == Fraction(2)
    assert result.intercept == Fraction(1)
    assert result.chi_square == Fraction(0)


def test_mismatched_uncertainty_units_converted():
    # 第三点以 mm 给出 y=6000mm、u=10mm（即 6m、0.01m），与全部用 m
    # 的同一组观测在有理数层面完全一致 → 证明值与不确定度都做了换算
    in_m = fit([
        P("0", "m", "1", "m", "0.01"),
        P("1", "m", "3", "m", "0.01"),
        P("2", "m", "6", "m", "0.01"),
    ], x_to="m", y_to="m")
    in_mm = fit([
        P("0", "m", "1", "m", "0.01"),
        P("1", "m", "3", "m", "0.01"),
        P("2", "m", "6000", "mm", "10"),
    ], x_to="m", y_to="m")
    assert in_mm == in_m


def test_sixteen_points_accepted():
    points = [P(str(i), y=str(2 * i + 1)) for i in range(16)]
    result = fit(points)
    assert result.slope == Fraction(2)
    assert result.intercept == Fraction(1)


def test_exponents_decimal_strings():
    # 指数记数法与等价的普通小数写法给出完全相同的精确结果
    points_exp = [
        P("1e2", "cm", "1e1", "mm", "10"),
        P("2e2", "cm", "3e1", "mm", "10"),
        P("3e2", "cm", "5e1", "mm", "10"),
    ]
    points_plain = [
        P("100", "cm", "10", "mm", "10"),
        P("200", "cm", "30", "mm", "10"),
        P("300", "cm", "50", "mm", "10"),
    ]
    assert fit(points_exp, x_to="m", y_to="m") == fit(points_plain, x_to="m", y_to="m")
    result = fit(points_plain, x_to="m", y_to="m")
    # x∈{1,2,3} m，y∈{0.01,0.03,0.05} m → y = 0.02·x − 0.01
    assert result.slope == Fraction(1, 50)
    assert result.intercept == Fraction(-1, 100)
    assert result.chi_square == Fraction(0)


# ---------------------------------------------------------------- 422 类错误

def test_all_x_equal():
    with pytest.raises(AllXEqualError):
        fit([P("1", y="1"), P("1", y="2"), P("1", y="3")])


def test_all_x_equal_after_conversion():
    # 100cm、1m、1000mm 换算到 m 后全相同
    with pytest.raises(AllXEqualError):
        fit([
            P("100", "cm", y="1"),
            P("1", "m", y="2"),
            P("1000", "mm", y="3"),
        ])


def test_two_distinct_x_but_repeated_is_fine():
    # 只要不全相同即可（x=0 两次、x=1 一次）：
    # 正规方程 Σw=3, Σwx=1, Σwy=5.2, Σwx²=1, Σwxy=3
    # a = (3·3 − 1·5.2)/(3·1 − 1) = 19/10
    result = fit([P("0", y="1"), P("0", y="1.2"), P("1", y="3")])
    assert result.slope == Fraction(19, 10)


def test_invalid_x_and_y():
    with pytest.raises(InvalidNumberError) as exc:
        fit([P(x="abc"), P("1"), P("2")])
    assert exc.value.extra["point"] == 0
    with pytest.raises(InvalidNumberError) as exc:
        fit([P("0"), P(y="nan"), P("2")])
    assert exc.value.extra["point"] == 1


def test_invalid_uncertainty_string():
    with pytest.raises(InvalidUncertaintyError) as exc:
        fit([P(y_uncertainty="x"), P("1"), P("2")])
    assert exc.value.extra["point"] == 0


@pytest.mark.parametrize("bad", ["0", "-0.1", "-1"])
def test_uncertainty_must_be_positive(bad):
    with pytest.raises(InvalidUncertaintyError):
        fit([P(y_uncertainty=bad), P("1"), P("2")])


def test_x_dimension_mismatch():
    with pytest.raises(PointDimensionMismatchError) as exc:
        fit([
            P("0", "m", y="1"),
            P("1", "s", y="3"),
            P("2", "m", y="5"),
        ], x_to="m", y_to="m")
    assert exc.value.extra["point"] == 1


def test_y_dimension_mismatch():
    with pytest.raises(PointDimensionMismatchError) as exc:
        fit([
            P("0", y="1", y_unit="m"),
            P("1", y="3", y_unit="kg"),
            P("2", y="5", y_unit="m"),
        ], x_to="m", y_to="m")
    assert exc.value.extra["point"] == 1


def test_dimensionless_target_dimensional_point():
    with pytest.raises(PointDimensionMismatchError):
        fit([P("0", "1", "1", "m"), P("1", "1", "3", "m"),
             P("2", "1", "5", "m")], x_to="1", y_to="1")


def test_offset_temperature_rejected_in_points():
    for unit in ("°C", "°F"):
        with pytest.raises(CalibrationUnitError) as exc:
            fit([P("0", unit, "1", "m"), P("1", unit, "3", "m"),
                 P("2", unit, "5", "m")], x_to="K", y_to="m")
        assert exc.value.error == "offset_temperature_not_allowed"
        assert exc.value.extra["point"] == 0


def test_offset_temperature_rejected_as_target():
    with pytest.raises(CalibrationUnitError) as exc:
        fit([P("0", "K", "1", "m"), P("1", "K", "3", "m"),
             P("2", "K", "5", "m")], x_to="°C", y_to="m")
    assert exc.value.error == "offset_temperature_not_allowed"
    assert exc.value.extra["field"] == "x_to"


def test_offset_in_compound_rejected():
    with pytest.raises(CalibrationUnitError) as exc:
        fit([P("0", "m/°C", "1", "m"), P("1", "m/°C", "3", "m"),
             P("2", "m/°C", "5", "m")], x_to="m/K", y_to="m")
    assert exc.value.error == "offset_temperature_not_allowed"


def test_unit_syntax_error_reports_point_and_position():
    with pytest.raises(CalibrationUnitError) as exc:
        fit([P("0", "m"), P("1", "m"),
             P("2", "kg&")], x_to="m", y_to="m")
    assert exc.value.error == "syntax_error"
    assert exc.value.extra["point"] == 2
    assert exc.value.extra["position"] == 2


def test_target_syntax_error_reports_field():
    with pytest.raises(CalibrationUnitError) as exc:
        fit([P("0"), P("1"), P("2")], x_to="m*", y_to="m")
    assert exc.value.error == "syntax_error"
    assert exc.value.extra == {"field": "x_to", "position": 2}
