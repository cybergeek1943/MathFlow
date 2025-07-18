# NOTE: all functions have a first arg of sympy Expr or Poly type. Also, Expr and Poly do contain these functions as methods.
from sympy.solvers.solvers import (
    solve,
    nsolve
)
from sympy.core.function import (
    expand_mul,
    expand_log,
    expand_func,
    expand_trig,
    expand_complex,
    expand_multinomial,
    expand_power_exp,
    expand_power_base,
    nfloat
)
from sympy.simplify.simplify import (
    separatevars,
    nthroot,
    hypersimp,
    logcombine,
)
from sympy.simplify.radsimp import (
    rad_rationalize,
    rcollect,
    collect_sqrt,
    collect_const,
)
from sympy.simplify.powsimp import (
    powdenest
)
from sympy.simplify.sqrtdenest import (
    sqrtdenest,
    is_sqrt,
    sqrt_depth
)
from sympy.core.exprtools import (
    gcd_terms,
    factor_terms,
    factor_nc
)
from sympy.polys.polyfuncs import (
    horner,
    symmetrize
)

# Define Aliases Here
convert_rationals_to_floats = nfloat


# Getter Functionality
__all__ = [n for n in dir() if not n.startswith('__') and not n.endswith('__')]  # Proxy can check this for available functions...
def get_attr(name):
    """Only returns name from namespace if it is in __all__. Otherwise, raise AttributeError."""
    if name in __all__:
        return globals()[name]
    raise AttributeError


# Testing
if __name__ == '__main__':
    # TODO write a test here to check weather Expr or Poly contains these functions as methods
    pass
