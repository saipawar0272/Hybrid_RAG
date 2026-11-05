"""Tools that the LLM may request. Weather is intentionally mocked."""
import ast
import operator as op
from langchain_core.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Exponent too large")
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only basic arithmetic expressions are allowed")


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, for example '125 * 8 + 40'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def get_weather_demo(city: str) -> str:
    """Return DEMO weather data for a city. This is mock data, not live weather."""
    return (
        f"DEMO DATA ONLY: Weather for {city}: 30°C and sunny. "
        "This result is simulated and must not be presented as live weather."
    )


TOOLS = [calculate, get_weather_demo]
