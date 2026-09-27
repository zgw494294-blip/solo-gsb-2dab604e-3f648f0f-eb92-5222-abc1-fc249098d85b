"""线性标定加权最小二乘拟合核心逻辑测试（不经过 HTTP）。"""

from fractions import Fraction

import pytest

from app.calibration import (
    CalibrationDimensionMismatch,
    CalibrationNumberError,
    CalibrationPoint,
    CalibrationUnitError,
    ConstantXError,
    fit_calibration,
)


def P(x="0", y="0", u="0.1", x_unit="m", y_unit="kg") -> CalibrationPoint:
    return CalibrationPoint(x=x, x_unit=x_unit, y=y, y_unit=y_unit,
                            y_uncertainty=u)


# ---------------------------------------------------------------- 成功拟合

def test_exact_line_three_points():
    # y = (1/30)·x + 1
    points = [P("0", "1"), P("30", "2"), P("60", "3")]
    result = fit_calibration("m", "kg", points)
    assert result.slope == Fraction(1, 30)
    assert result.intercept == Fraction(1)
    assert result.chi_square == Fraction(0)


def test_unweighted_least_squares():
    # 等权即普通最小二乘：y = x 附近的 (0,0),(1,2),(2,2)
    # a = 1, b = 1/3
    points = [P("0", "0"), P("1", "2"), P("2", "2")]
    result = fit_calibration("m", "kg", points)
    assert result.slope == Fraction(1)
    assert result.intercept == Fraction(1, 3)
    # 残差 (-1/3, 2/3, -1/3)/0.1 → chi² = (1+4+1)·100/9 = 200/3
    assert result.chi_square == Fraction(200, 3)


def test_weights_inverse_square_uncertainty():
    # 中间点不确定度大一倍、权为 1/4：(0,1,u=.1),(1,2.1,u=.2),(2,2.9,u=.1)
    points = [P("0", "1.0", "0.1", "1", "1"),
              P("1", "2.1", "0.2", "1", "1"),
              P("2", "2.9", "0.1", "1", "1")]
    result = fit_calibration("1", "1", points)
    assert result.slope == Fraction(19, 20)       # 0.95
    assert result.intercept == Fraction(61, 60)  # 1.01666…
    assert result.chi_square == Fraction(1, 2)


def test_units_converted_to_targets():
    # x 以 cm 给、y 以 g 给，目标 m / kg，结果应与直接用 m / kg 相同
    points = [P("0", "1000", "100", "cm", "g"),
              P("3000", "2000", "100", "cm", "g"),
              P("6000", "3000", "100", "cm", "g")]
    result = fit_calibration("m", "kg", points)
    assert result.slope == Fraction(1, 30)
    assert result.intercept == Fraction(1)
    assert result.chi_square == Fraction(0)


def test_compound_units():
    # x 时间 h，y 长度 km，目标 s / m：y = 10 km/h · x，起点 0
    points = [P("0", "0", "0.01", "h", "km"),
              P("1", "10", "0.01", "h", "km"),
              P("2", "20", "0.01", "h", "km")]
    result = fit_calibration("s", "m", points)
    # 10 km/h = 10000 m / 3600 s = 25/9 m/s
    assert result.slope == Fraction(25, 9)
    assert result.intercept == Fraction(0)


def test_uncertainty_scales_with_y_unit():
    # 用 g 给出、u=100 g，换到 kg 后 u=0.1 kg，与直接给 kg 等价
    points_g = [P("0", "0", "100", "m", "g"),
                P("30", "1000", "100", "m", "g"),
                P("60", "2000", "100", "m", "g")]
    points_kg = [P("0", "0", "0.1"),
                 P("30", "1", "0.1"),
                 P("60", "2", "0.1")]
    assert fit_calibration("m", "kg", points_g) == fit_calibration(
        "m", "kg", points_kg)


def test_dimensionless_one_units():
    points = [P("1", "2", "0.1", "1", "1"),
              P("2", "4", "0.1", "1", "1"),
              P("3", "6", "0.1", "1", "1")]
    result = fit_calibration("1", "1", points)
    assert result.slope == Fraction(2)
    assert result.intercept == Fraction(0)


def test_kelvin_ok():
    points = [P("0", "300", "1", "1", "K"),
              P("1", "310", "1", "1", "K"),
              P("2", "320", "1", "1", "K")]
    result = fit_calibration("1", "K", points)
    assert result.slope == Fraction(10)
    assert result.intercept == Fraction(300)


def test_sixteen_points():
    points = [P(str(i), str(2 * i), "1") for i in range(16)]
    result = fit_calibration("m", "kg", points)
    assert result.slope == Fraction(2)
    assert result.intercept == Fraction(0)
    assert result.chi_square == Fraction(0)


def test_negative_x_and_y():
    points = [P("-1", "-2", "0.1", "1", "1"),
              P("0", "0", "0.1", "1", "1"),
              P("1", "2", "0.1", "1", "1")]
    result = fit_calibration("1", "1", points)
    assert result.slope == Fraction(2)
    assert result.intercept == Fraction(0)


def test_exact_decimals_no_rounding_error():
    # 0.1 这类十进制输入应被精确处理：直线上的点 chi² 恰为 0
    points = [P("0.1", "0.2", "0.1", "1", "1"),
              P("0.2", "0.4", "0.1", "1", "1"),
              P("0.3", "0.6", "0.1", "1", "1")]
    result = fit_calibration("1", "1", points)
    assert result.chi_square == Fraction(0)
    assert result.slope == Fraction(2)


# ---------------------------------------------------------------- 422：x 全同

def test_constant_x_rejected():
    with pytest.raises(ConstantXError):
        fit_calibration("m", "kg", [P("1", "1"), P("1", "2"), P("1", "3")])


def test_constant_x_after_unit_conversion_rejected():
    # 60000 cm == 600 m == 0.6 km，写法不同但换算后完全相同
    points = [P("60000", "1", "0.1", "cm", "kg"),
              P("600", "2", "0.1", "m", "kg"),
              P("0.6", "3", "0.1", "km", "kg")]
    with pytest.raises(ConstantXError):
        fit_calibration("m", "kg", points)


# ---------------------------------------------------------------- 422：数值

def test_invalid_values():
    for field, kwargs in (("x", dict(x="abc")), ("y", dict(y="nan"))):
        with pytest.raises(CalibrationNumberError) as exc_info:
            fit_calibration("m", "kg",
                            [P(**kwargs), P("1", "2"), P("2", "3")])
        assert exc_info.value.error == "invalid_value"
        assert exc_info.value.extra["point"] == 0
        assert exc_info.value.extra["field"] == field

    # 不确定度字符串本身非法（nan / inf / 非数字）归入 invalid_uncertainty
    for bad in ("inf", "nan", "abc"):
        with pytest.raises(CalibrationNumberError) as exc_info:
            fit_calibration("m", "kg",
                            [P(u=bad), P("1", "2"), P("2", "3")])
        assert exc_info.value.error == "invalid_uncertainty"
        assert exc_info.value.extra == {"point": 0, "field": "y_uncertainty"}


def test_uncertainty_must_be_positive():
    for bad in ("0", "-0.1", "-1"):
        with pytest.raises(CalibrationNumberError) as exc_info:
            fit_calibration("m", "kg",
                            [P("0", "1", bad), P("1", "2"), P("2", "3")])
        assert exc_info.value.error == "invalid_uncertainty"
        assert exc_info.value.extra == {"point": 0, "field": "y_uncertainty"}


def test_point_index_reported():
    with pytest.raises(CalibrationNumberError) as exc_info:
        fit_calibration("m", "kg",
                        [P("0", "1"), P("x", "2"), P("2", "3")])
    assert exc_info.value.extra["point"] == 1


# ---------------------------------------------------------------- 422：单位与量纲

def test_dimension_mismatch_x():
    with pytest.raises(CalibrationDimensionMismatch) as exc_info:
        fit_calibration("m", "kg",
                        [P("0", "1", x_unit="s"), P("1", "2"), P("2", "3")])
    assert exc_info.value.extra == {"point": 0, "field": "x_unit"}


def test_dimension_mismatch_y():
    with pytest.raises(CalibrationDimensionMismatch) as exc_info:
        fit_calibration("1", "1",
                        [P("0", "1", y_unit="m", x_unit="1"),
                         P("1", "2", x_unit="1", y_unit="1"),
                         P("2", "3", x_unit="1", y_unit="1")])
    assert exc_info.value.extra == {"point": 0, "field": "y_unit"}


def test_target_dimension_mismatch():
    # 点为 m，x_to 为 s
    with pytest.raises(CalibrationDimensionMismatch) as exc_info:
        fit_calibration("s", "kg",
                        [P("0", "1"), P("1", "2"), P("2", "3")])
    assert exc_info.value.extra == {"point": 0, "field": "x_unit"}


def test_offset_temperature_rejected():
    # 目标单位
    with pytest.raises(CalibrationUnitError) as exc_info:
        fit_calibration("°C", "kg", [P("0", "1"), P("1", "2"), P("2", "3")])
    assert exc_info.value.error == "offset_temperature_not_allowed"
    assert exc_info.value.extra == {"position": 0, "field": "x_to"}

    # 点单位
    with pytest.raises(CalibrationUnitError) as exc_info:
        fit_calibration("K", "kg",
                        [P("0", "1", x_unit="°F"), P("1", "2"), P("2", "3")])
    assert exc_info.value.error == "offset_temperature_not_allowed"
    assert exc_info.value.extra == {"position": 0, "point": 0, "field": "x_unit"}


def test_unit_syntax_error_has_coordinates():
    with pytest.raises(CalibrationUnitError) as exc_info:
        fit_calibration("m&", "kg", [P("0", "1"), P("1", "2"), P("2", "3")])
    assert exc_info.value.error == "syntax_error"
    assert exc_info.value.extra == {"position": 1, "field": "x_to"}

    with pytest.raises(CalibrationUnitError) as exc_info:
        fit_calibration("m", "kg",
                        [P("0", "1"), P("1", "2", y_unit="kg&"), P("2", "3")])
    assert exc_info.value.error == "syntax_error"
    assert exc_info.value.extra == {"position": 2, "point": 1, "field": "y_unit"}
