"""POST /fit-calibration 的 HTTP API 测试。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def fit(x_to, y_to, points):
    return client.post("/fit-calibration",
                       json={"x_to": x_to, "y_to": y_to, "points": points})


def pt(x="0", y="0", u="0.1", x_unit="m", y_unit="kg"):
    return {"x": x, "x_unit": x_unit, "y": y, "y_unit": y_unit,
            "y_uncertainty": u}


# ---------------------------------------------------------------- 成功

def test_exact_line():
    r = fit("m", "kg", [pt("0", "1"), pt("30", "2"), pt("60", "3")])
    assert r.status_code == 200
    assert r.json() == {"slope": "0.033333333333",
                        "intercept": "1",
                        "chi_square": "0"}
    for key in ("slope", "intercept", "chi_square"):
        assert isinstance(r.json()[key], str)


def test_integer_results():
    # y = 2x
    r = fit("m", "kg", [pt("0", "0"), pt("1", "2"), pt("2", "4")])
    assert r.status_code == 200
    assert r.json() == {"slope": "2", "intercept": "0", "chi_square": "0"}


def test_weighted_fit():
    r = fit("1", "1", [
        pt("0", "1.0", "0.1", "1", "1"),
        pt("1", "2.1", "0.2", "1", "1"),
        pt("2", "2.9", "0.1", "1", "1"),
    ])
    assert r.status_code == 200
    assert r.json() == {"slope": "0.95",
                        "intercept": "1.016666666667",
                        "chi_square": "0.5"}


def test_point_units_converted():
    # x 用 cm、y 用 g，目标 m / kg
    r = fit("m", "kg", [
        pt("0", "1000", "100", "cm", "g"),
        pt("3000", "2000", "100", "cm", "g"),
        pt("6000", "3000", "100", "cm", "g"),
    ])
    assert r.status_code == 200
    assert r.json()["slope"] == "0.033333333333"
    assert r.json()["intercept"] == "1"
    assert r.json()["chi_square"] == "0"


def test_compound_units():
    # y = 10 km/h · x，目标 s / m → slope = 25/9 m/s
    r = fit("s", "m", [
        pt("0", "0", "0.01", "h", "km"),
        pt("1", "10", "0.01", "h", "km"),
        pt("2", "20", "0.01", "h", "km"),
    ])
    assert r.status_code == 200
    assert r.json()["slope"] == "2.777777777778"
    assert r.json()["intercept"] == "0"


def test_dimensionless_one_units():
    r = fit("1", "1", [
        pt("1", "2", "0.1", "1", "1"),
        pt("2", "4", "0.1", "1", "1"),
        pt("3", "6", "0.1", "1", "1"),
    ])
    assert r.status_code == 200
    assert r.json()["slope"] == "2"


def test_kelvin_ok():
    r = fit("1", "K", [
        pt("0", "300", "1", "1", "K"),
        pt("1", "310", "1", "1", "K"),
        pt("2", "320", "1", "1", "K"),
    ])
    assert r.status_code == 200
    assert r.json() == {"slope": "10", "intercept": "300", "chi_square": "0"}


def test_rounding_half_even_tie():
    # slope = 1/3 → 0.333333333333；用 1/30 构造 0.0333333333333… 场景
    r = fit("1", "1", [
        pt("0", "0", "1", "1", "1"),
        pt("3", "1", "1", "1", "1"),
        pt("6", "2", "1", "1", "1"),
    ])
    assert r.status_code == 200
    assert r.json()["slope"] == "0.333333333333"
    assert r.json()["intercept"] == "0"
    # 点 (0,0),(3,1),(6,2) 恰在 y=x/3 上
    assert r.json()["chi_square"] == "0"


def test_sixteen_points():
    points = [pt(str(i), str(2 * i), "1") for i in range(16)]
    r = fit("m", "kg", points)
    assert r.status_code == 200
    assert r.json()["slope"] == "2"
    assert r.json()["intercept"] == "0"


def test_scientific_notation_values():
    r = fit("1", "1", [
        pt("1e0", "2e0", "1e-1", "1", "1"),
        pt("2e0", "4.0e0", "1e-1", "1", "1"),
        pt("3e0", "6.0", ".1", "1", "1"),
    ])
    assert r.status_code == 200
    assert r.json()["slope"] == "2"


# ---------------------------------------------------------------- 422：x 全同

def test_constant_x():
    r = fit("m", "kg", [pt("1", "1"), pt("1", "2"), pt("1", "3")])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "constant_x_values"


def test_constant_x_after_conversion():
    r = fit("m", "kg", [
        pt("60000", "1", "0.1", "cm", "kg"),
        pt("600", "2", "0.1", "m", "kg"),
        pt("0.6", "3", "0.1", "km", "kg"),
    ])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "constant_x_values"


# ---------------------------------------------------------------- 422：数值

def test_invalid_numbers():
    bad_payloads = (
        {"x": "abc"},
        {"y": "nan"},
        {"y": "inf"},
        {"y_uncertainty": "xyz"},
        {"x": "1e10000"},
    )
    good = [pt("0", "1"), pt("1", "2"), pt("2", "3")]
    for bad in bad_payloads:
        points = [dict(good[0], **bad)] + good[1:]
        r = fit("m", "kg", points)
        assert r.status_code == 422, bad
        detail = r.json()["detail"]
        assert detail["error"] in ("invalid_value", "invalid_uncertainty")
        assert detail["point"] == 0


def test_uncertainty_must_be_positive():
    for bad in ("0", "-0.1", "-1"):
        r = fit("m", "kg",
                [pt("0", "1", bad), pt("1", "2"), pt("2", "3")])
        assert r.status_code == 422, bad
        detail = r.json()["detail"]
        assert detail["error"] == "invalid_uncertainty"
        assert detail["point"] == 0
        assert detail["field"] == "y_uncertainty"


def test_error_point_index():
    r = fit("m", "kg",
            [pt("0", "1"), pt("1", "x"), pt("2", "3")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "invalid_value"
    assert detail["point"] == 1
    assert detail["field"] == "y"


# ---------------------------------------------------------------- 422：单位与量纲

def test_dimension_mismatch():
    r = fit("m", "kg", [
        pt("0", "1", x_unit="s"), pt("1", "2"), pt("2", "3")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "dimension_mismatch"
    assert detail["point"] == 0
    assert detail["field"] == "x_unit"


def test_dimension_mismatch_y():
    r = fit("1", "1", [
        pt("0", "1", y_unit="m", x_unit="1"),
        pt("1", "2", x_unit="1", y_unit="1"),
        pt("2", "3", x_unit="1", y_unit="1")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "dimension_mismatch"
    assert detail["field"] == "y_unit"


def test_offset_temperature_target():
    for target in ("°C", "°F"):
        r = fit(target, "kg", [pt("0", "1"), pt("1", "2"), pt("2", "3")])
        assert r.status_code == 422, target
        detail = r.json()["detail"]
        assert detail["error"] == "offset_temperature_not_allowed"
        assert detail["field"] == "x_to"


def test_offset_temperature_point():
    for unit in ("°C", "°F", "K*°C"):
        r = fit("K", "kg", [
            pt("0", "1", x_unit=unit), pt("1", "2"), pt("2", "3")])
        assert r.status_code == 422, unit
        detail = r.json()["detail"]
        assert detail["error"] == "offset_temperature_not_allowed"
        assert detail["point"] == 0
        assert detail["field"] == "x_unit"


def test_unit_syntax_error_position():
    r = fit("m&", "kg", [pt("0", "1"), pt("1", "2"), pt("2", "3")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["field"] == "x_to"
    assert detail["position"] == 1

    r = fit("m", "kg", [
        pt("0", "1"), pt("1", "2", y_unit="kg&"), pt("2", "3")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["point"] == 1
    assert detail["field"] == "y_unit"
    assert detail["position"] == 2


# ---------------------------------------------------------------- 422：请求体结构

def test_missing_fields_422():
    assert client.post("/fit-calibration", json={}).status_code == 422
    assert client.post("/fit-calibration",
                       json={"x_to": "m", "y_to": "kg"}).status_code == 422
    assert client.post("/fit-calibration",
                       json={"x_to": "m", "points": []}).status_code == 422


def test_point_count_bounds():
    three = [pt("0", "1"), pt("1", "2"), pt("2", "3")]
    assert client.post("/fit-calibration", json={
        "x_to": "m", "y_to": "kg",
        "points": three[:2]}).status_code == 422
    seventeen = [pt(str(i), str(i), "1") for i in range(17)]
    assert fit("m", "kg", seventeen).status_code == 422


def test_missing_point_fields():
    base = pt("0", "1")
    for key in ("x", "x_unit", "y", "y_unit", "y_uncertainty"):
        bad = {k: v for k, v in base.items() if k != key}
        r = fit("m", "kg",
                [bad, pt("1", "2"), pt("2", "3")])
        assert r.status_code == 422, key


def test_unknown_field_rejected():
    r = client.post("/fit-calibration", json={
        "x_to": "m", "y_to": "kg",
        "points": [dict(pt("0", "1"), note="nope"),
                   pt("1", "2"), pt("2", "3")]})
    assert r.status_code == 422
    r = client.post("/fit-calibration", json={
        "x_to": "m", "y_to": "kg",
        "points": [pt("0", "1"), pt("1", "2"), pt("2", "3")],
        "note": "nope"})
    assert r.status_code == 422
