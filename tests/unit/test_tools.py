
import pytest

from src.tools import calculate, get_weather_demo


# --------------------------------------------------
# CALCULATOR TOOL TESTS
# --------------------------------------------------

@pytest.mark.parametrize(
    "expression, expected",
    [
        ("10 + 5", "15"),
        ("20 - 5", "15"),
        ("10 * 5", "50"),
        ("20 / 5", "4.0"),
        ("10 % 3", "1"),
        ("2 ** 3", "8"),
        ("-10", "-10"),
        ("+10", "10"),
        ("(10 + 5) * 2", "30"),
    ],
)
def test_calculate_valid_expressions(expression, expected):
    result = calculate.invoke({"expression": expression})

    assert result == expected


@pytest.mark.parametrize(
    "expression",
    [
        "10 +",
        "abc",
        "10 + 'hello'",
        "print(10)",
        "__import__('os').system('dir')",
        "10 ** 9",
    ],
)
def test_calculate_rejects_invalid_expressions(expression):
    result = calculate.invoke({"expression": expression})

    assert result.startswith("Calculation error:")


def test_calculate_handles_division_by_zero():
    result = calculate.invoke({"expression": "10 / 0"})

    assert result.startswith("Calculation error:")
    assert "division by zero" in result.lower()


def test_calculate_rejects_boolean_expression():
    result = calculate.invoke({"expression": "True"})

    # Current implementation accepts True as 1 because bool is a subclass of int.
    # This test documents a security/validation gap to fix next.
    assert result.startswith("Calculation error:")
    assert "Only basic arithmetic expressions are allowed" in result


# --------------------------------------------------
# WEATHER TOOL TESTS
# --------------------------------------------------

def test_get_weather_demo_returns_mock_weather():
    result = get_weather_demo.invoke({"city": "Pune"})

    assert "Pune" in result
    assert "30" in result
    assert "sunny" in result.lower()


def test_get_weather_demo_identifies_data_as_simulated():
    result = get_weather_demo.invoke({"city": "Mumbai"})

    assert "DEMO DATA ONLY" in result
    assert "simulated" in result.lower()
    assert "not be presented as live weather" in result
