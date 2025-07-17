# NOTE: all of these functions have a first argument of `func` (a callable). This way they can be wrapped up into a method of a numerical proxy
# noinspection PyUnresolvedReferences
from scipy.differentiate import (
    derivative,
    jacobian,
    hessian,
)

# noinspection PyUnresolvedReferences
from scipy.integrate import (
    # Definite Integrations
    quad,
    quad_vec,
    cubature,
    dblquad,
    tplquad,
    nquad,
    tanhsinh,
    fixed_quad,
    qmc_quad,

    # Sums
    nsum,
)

# noinspection PyUnresolvedReferences
from scipy.optimize import (
    # Local Optimizers
    minimize_scalar,
    minimize,

    # Global Optimizers
    basinhopping,
    brute,
    differential_evolution,
    shgo,
    dual_annealing,
    direct,

    # Least Squares
    least_squares,
    curve_fit,

    # Root Finding
    # - For Scaler Functions
    root_scalar,
    brentq,
    brenth,
    ridder,
    bisect,
    newton,
    toms748,
    # - Fixed point
    fixed_point,
    # - For Vector Functions
    root
)

# noinspection PyUnresolvedReferences
from scipy.optimize.elementwise import (
    # Root Finding
    find_root,
    bracket_root,

    # Minimization
    find_minimum,
    bracket_minimum,
)


# Define Aliases Here
solve = root_scalar


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
