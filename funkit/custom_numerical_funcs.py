import numpy as np
from numpy import linspace
from numpy.polynomial import Polynomial
from scipy.optimize import root
from scipy.differentiate import derivative
from typing import Callable, Sequence


# ================================ Calculus ================================
def derivative_lambda(func: Callable, *args, df_order: int = 1, **kwargs) -> Callable:
    """Returns a lambda function which is the numerical derivative of self."""
    def nth_derivative(x):
        current_func = func
        for _ in range(df_order):
            current_func = lambda y, f=current_func: derivative(f, y, *args, **kwargs).df
        return current_func(x)
    return nth_derivative


# ================================ Roots ================================
def all_roots(func: Callable,
              args: tuple = (),
              x0s: Sequence[float] = None,  # if no guesses, bounds are used to generate guesses
              bounds: tuple[float, float] = (-10, 10),
              resolution: int = 100,
              dup_tol: float = 1e-4,
              method: str = 'hybr',
              tol: float = 1e-8,
              **kwargs) -> np.ndarray:
    roots: list[float] = []

    if not x0s:
        a, b = bounds
        x0s = linspace(a, b, resolution)  # create linespace search space for roots
    else:
        a, b = min(x0s), max(x0s)

    for x in x0s:
        try:
            # noinspection PyTypeChecker
            if (sol:=root(fun=func,
                          x0=x,
                          args=args,
                          method=method,
                          tol=tol, **kwargs)).success:
                root_val: float = sol.x[0]

                if not np.isfinite(root_val) or any(abs(root_val - u) < dup_tol for u in roots):
                    continue  # skip infinite's and duplicates
                if a <= root_val <= b:
                    roots.append(root_val)
        except:
            pass

    return np.array(roots)


best_kwargs_for_polynomials_in_all_roots_func: dict = {
    'resolution': 400,  # Higher resolution (polys can have more roots)
    'method': 'lm',  # use Levenberg-Marquardt method because polys are smooth (can be more efficient than 'hybr')
    'tol': 1e-10,  # lower tolerance because polys have fewer issues with high accuracy.
}


def all_poly_roots(coeffs: Sequence[float], cleaning_tol: float = 1e-8) -> np.ndarray:
    if len(coeffs) <= 1:
        return np.array([])
    roots: np.ndarray = Polynomial(coeffs[::-1]).roots()  # Find roots (uses eigenvalues)
    roots = np.where(np.abs(roots.imag) < cleaning_tol, roots.real, roots)  # Clean out small imaginary noise
    return np.array(sorted(roots))


# ================================ Testing ================================
if __name__ == "__main__":
    # Polynomial (uses default hybr method)
    roots1 = all_roots(lambda x: x**3 - 2*x, bounds=(-3, 5))
    print(roots1)

    # # Transcendental
    # roots2 = all_roots(lambda x: np.sin(x) - 0.5 * x, bounds=(-10, 10))
    # print(f"Transcendental roots: {roots2}")
    #
    # # Exponential
    # roots3 = all_roots(lambda x: np.exp(x) - 2 * x - 1, bounds=(-2, 3))
    # print(f"Exponential roots: {roots3}")
    #
    # # Function that touches zero without crossing (x²)
    # roots4 = all_roots(lambda x: x ** 2, bounds=(-2, 2))
    # print(f"x² roots: {roots4}")
    #
    # # Function with multiple roots
    # roots5 = all_roots(lambda x: x ** 4 - 10 * x ** 2 + 9, bounds=(-4, 4))
    # print(f"x⁴ - 10x² + 9 roots: {roots5}")