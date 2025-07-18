import sympy as sp
import re
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.parsing.latex import parse_latex


def _is_latex(expr: str) -> bool:
    """Check if expression contains LaTeX indicators.

    >>> _is_latex(r'\\frac{x^2 + 1}{x - 1}')
    True
    >>> _is_latex(r'\\sin(x) + \\cos(x)')
    True
    >>> _is_latex(r'2 \\prod 3')
    True
    >>> _is_latex(r'2 + 3')
    False
    """
    return bool(re.search(r'\\[a-zA-Z]+', expr))


def _preprocess_expression(expr: str) -> str:
    """Convert natural notation to SymPy format: x^2 → x**2, ln → log, |x| → Abs(x)

    Examples:
    >>> _preprocess_expression("2x^2 + 3x + 1")
    '2x**2 + 3x + 1'
    >>> _preprocess_expression("|2x^2| + |3x + 1|")
    'Abs(2x**2) + Abs(3x + 1)'
    >>> _preprocess_expression("|2x^2| + ln(|3x| + 1)")
    'Abs(2x**2) + log(Abs(3x) + 1)'
    """
    symbols: dict[str, str] = {
        '^': '**',
        '•': '*',
        'ln(': 'log(',
    }
    for symbol in symbols:
        expr = expr.replace(symbol, symbols[symbol])
    return re.sub(r'\|([^|]+)\|', r'Abs(\1)', expr)


def parse_expression_str(expr: str, **kwargs) -> sp.Expr:
    """
    Elegantly parse string mathematical expressions with automatic format detection.

    Features:
    - Auto-detects LaTeX vs standard notation
    - Implicit multiplication: "2x" → "2*x"
    - Natural log: "ln(x)" → "log(x)"
    - Absolute value: "|x-1|" → "Abs(x-1)"
    - Power notation: "x^2" → "x**2"
    - Function notation: "sin x" → "sin(x)"

    Args:
        expr: String mathematical expression

    Returns:
        SymPy expression

    Examples:
        >>> parse_expression_str("2x^2 + 3x + 1")
        2*x**2 + 3*x + 1
        >>> parse_expression_str("|x - 1| + ln(x)")
        log(x) + Abs(x - 1)
        >>> parse_expression_str(r"\\frac{x^2+1}{x-1}")
        (x**2 + 1)/(x - 1)
    """

    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("Expression must be a non-empty string")

    expr = expr.strip()

    # Try LaTeX first if LaTeX indicators are present
    if _is_latex(expr):
        try:
            return parse_latex(expr)
        except:
            pass  # Fall through to standard parsing

    # Preprocess for natural mathematical notation
    preprocessed = _preprocess_expression(expr)

    # Parse with SymPy transformations for implicit multiplication
    transformations = standard_transformations + (implicit_multiplication_application,)

    try:
        return parse_expr(preprocessed, transformations=transformations, **kwargs)
    except Exception as e:
        raise ValueError(f"Could not parse expression '{expr}': {e}")


def validate_expression(expr: str) -> bool:
    """Check if expression can be parsed without raising an exception."""
    try:
        parse_expression_str(expr)
        return True
    except:
        return False


# Examples and demonstrations
if __name__ == "__main__":
    print(isinstance(parse_expression_str("2x^2"), sp.Expr))
    exit()

    # ---- Testing ----
    # noinspection PyUnreachableCode
    test_cases = [
        # Basic arithmetic with implicit multiplication
        "2x + 3y",
        "2x^2 + 3x + 1",
        "x(x + 1)",
        "(x + 1)(x - 1)",

        # Functions
        "sin(x) + cos(x)",
        "ln(x) + exp(x)",
        "sqrt(x^2 + y^2)",

        # Absolute values
        "|x|",
        "|x - 1|",
        "|sin(x)|",

        # Mixed complexity
        "2x|x - 1| + ln(x)",
        "e^(x^2) + ln|x|",

        # LaTeX
        r"\frac{x^2 + 1}{x - 1}",
        r"\sin(x) + \cos(x)",
        r"\sqrt{x^2 + y^2}",
        r"x \times y",

        # Edge cases
        "x + y",  # Simple
        "xyz",  # Multiple variables
        "2(x + 1)3(y - 1)",  # Complex implicit multiplication
    ]

    print("Testing Expression Parser")
    print("=" * 50)

    for expr in test_cases:
        try:
            result = parse_expression_str(expr)
            print(f"✓ '{expr}' → {result}")
        except Exception as e:
            print(f"✗ '{expr}' → Error: {e}")
