# NOTE: all functions have a first arg of sympy Expr or Poly type. Also, Expr and Poly do contain these functions as methods.
from sympy.printing import (
    pretty,
    ccode,
    cxxcode,
    rcode,
    fcode,
    smtlib_code,
    mathematica_code,
    maple_code,
    jscode,
    julia_code,
    octave_code,
    rust_code,
    latex,
    mathml,
    pycode,
    python
)
from sympy.printing.tree import tree


# Define Aliases Here
pass


# Getter Functionality
__all__ = [n for n in dir() if not n.startswith('__') and not n.endswith('__')]  # Proxy can check this for available functions...
def get_attr(name):
    """Only returns name from namespace if it is in __all__. Otherwise, raise AttributeError."""
    if name in __all__:
        return globals()[name]
    raise AttributeError


# Testing
if __name__ == '__main__':
    pass
