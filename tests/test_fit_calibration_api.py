"""POST /fit-calibration 的 HTTP API 测试。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def fit(points, x_to="m", y_to="m"):
    return client.post("/fit-calibration",
                       json={"x_to": x_to, "y_to": y_to, "points": points})


def p(x="0", x_unit="m", y="0", y_unit="m", y_uncertainty="1"):
    return {"x": x, "x_unit": x_unit, "y": y, "y_unit": y_unit,
            "y_uncertainty": y_uncertainty}


# ---------------------------------------------------------------- 成功

def test_exact_line():
    r = fit([p("0", y="1"), p("1", y="3"), p("2", y="5")])
    assert r.status_code == 200
    assert r.json() == {"slope": "2", "intercept": "1", "chi_square": "0"}
    for value in r.json().values():
        assert isinstance(value, str)


def test_mixed_units():
    # x 用 cm、y 用 mm，目标 m：y[m] = 2·x[m] − 0.01
    r = fit([
        p("1", "cm", "10", "mm", "10"),
        p("2", "cm", "30", "mm", "10"),
        p("3", "cm", "50", "mm", "10"),
    ], x_to="m", y_to="m")
    assert r.status_code == 200
    assert r.json() == {"slope": "2", "intercept": "-0.01", "chi_square": "0"}


def test_dimensionless():
    r = fit([
        p("0", "1", "1", "1", "0.5"),
        p("1", "1", "3", "1", "0.5"),
        p("2", "1", "5", "1", "0.5"),
    ], x_to="1", y_to="1")
    assert r.status_code == 200
    assert r.json() == {"slope": "2", "intercept": "1", "chi_square": "0"}


def test_chi_square_value():
    # 等权三点 (0,1.1),(1,2.9),(2,5.2)：χ² = 1/24 = 0.041666…
    r = fit([p("0", y="1.1"), p("1", y="2.9"), p("2", y="5.2")])
    assert r.status_code == 200
    body = r.json()
    assert body["slope"] == "2.05"
    assert body["intercept"] == "1.016666666667"
    assert body["chi_square"] == "0.041666666667"


def test_kelvin_ok_but_celsius_rejected():
    r = fit([
        p("300", "K", "1", "m", "0.1"),
        p("310", "K", "2", "m", "0.1"),
        p("320", "K", "3", "m", "0.1"),
    ], x_to="K", y_to="m")
    assert r.status_code == 200
    assert r.json()["slope"] == "0.1"

    r = fit([
        p("300", "°C", "1", "m", "0.1"),
        p("310", "°C", "2", "m", "0.1"),
        p("320", "°C", "3", "m", "0.1"),
    ], x_to="K", y_to="m")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "offset_temperature_not_allowed"
    assert detail["point"] == 0


def test_sixteen_points_bound():
    body = [p(str(i), y=str(2 * i + 1)) for i in range(16)]
    assert fit(body).status_code == 200


# ---------------------------------------------------------------- 422

def test_all_x_equal():
    r = fit([p("1", y="1"), p("1", y="2"), p("1", y="3")])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "all_x_equal"


def test_all_x_equal_after_conversion():
    r = fit([
        p("100", "cm", y="1"),
        p("1", "m", y="2"),
        p("1000", "mm", y="3"),
    ])
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "all_x_equal"


def test_invalid_x():
    r = fit([p(x="nan"), p("1"), p("2")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "invalid_value"
    assert detail["point"] == 0


def test_invalid_uncertainty():
    for bad in ("abc", "0", "-0.5", "inf"):
        r = fit([p(y_uncertainty=bad), p("1"), p("2")])
        assert r.status_code == 422, bad
        detail = r.json()["detail"]
        assert detail["error"] == "invalid_uncertainty"
        assert detail["point"] == 0


def test_dimension_mismatch_has_point():
    r = fit([
        p("0", "m", "1", "m"),
        p("1", "s", "3", "m"),
        p("2", "m", "5", "m"),
    ])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "dimension_mismatch"
    assert detail["point"] == 1


def test_unit_syntax_error_has_point_and_position():
    r = fit([p("0"), p("1"), p("2", "m", y="5", y_unit="kg&")])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["point"] == 2
    assert detail["position"] == 2


def test_target_syntax_error_has_field():
    r = fit([p("0"), p("1"), p("2")], x_to="m/", y_to="m")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["field"] == "x_to"
    assert detail["position"] == 2


def test_target_offset_temperature_rejected():
    r = fit([
        p("300", "K", "1", "m", "0.1"),
        p("310", "K", "2", "m", "0.1"),
        p("320", "K", "3", "m", "0.1"),
    ], x_to="K", y_to="°F")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "offset_temperature_not_allowed"
    assert detail["field"] == "y_to"


# ---------------------------------------------------------------- 422：请求体结构

def test_missing_fields_422():
    assert client.post("/fit-calibration", json={}).status_code == 422
    assert client.post("/fit-calibration",
                       json={"y_to": "m", "points": []}).status_code == 422
    assert client.post("/fit-calibration",
                       json={"x_to": "m", "y_to": "m"}).status_code == 422


def test_point_count_bounds():
    two = [p(str(i), y=str(i)) for i in range(2)]
    seventeen = [p(str(i), y=str(i)) for i in range(17)]
    assert fit(two).status_code == 422
    assert fit(seventeen).status_code == 422


def test_missing_point_fields():
    base = p()
    for field in ("x", "x_unit", "y", "y_unit", "y_uncertainty"):
        broken = {k: v for k, v in base.items() if k != field}
        r = fit([broken, p("1"), p("2")])
        assert r.status_code == 422, field


def test_unknown_field_rejected():
    point = p()
    point["note"] = "nope"
    assert fit([point, p("1"), p("2")]).status_code == 422
    assert client.post("/fit-calibration", json={
        "x_to": "m", "y_to": "m",
        "points": [p(), p("1"), p("2")], "extra": 1}).status_code == 422
