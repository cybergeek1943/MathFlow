from sympy import (
    factorial, diff, linsolve, zeros, lambdify, expand_multinomial, lcm_list
)
# Import Sympy Types
from sympy import (
    Expr, Symbol, Add, Matrix, Number, Rational, Integer, Float, FiniteSet, Tuple
)
from itertools import zip_longest
from tabulate import tabulate  # for the verbose functions
from typing import Callable, Literal, Sequence, Any
from mpmath.ctx_mp_python import mpnumeric
import mpmath


def _print_multiplication_table(l_coeffs, t_coeffs):
    rows: list = []
    for lc in l_coeffs:
        row = [lc]
        for tc in t_coeffs:
            if lc == '1':
                cell_value = str(tc)
            else:
                if tc == 1:
                    cell_value = lc
                elif str(tc).startswith('-'):
                    cell_value = f'{lc}*({tc})'
                else:
                    cell_value = f'{lc}*{tc}'
            row.append(cell_value)
        rows.append(row)
    print(tabulate(rows, headers=t_coeffs, tablefmt='rounded_grid'))


# ================================ Universal Helper Functions ================================
def taylor_coeffs(expr: Expr,
                  x: Symbol,
                  x0: float,
                  order: int) -> list[Number]:
    coeffs: list[Number] = []
    current_expr: Expr = expr
    for i in range(order + 1):
        # noinspection PyTypeChecker
        coeffs.append(current_expr.subs(x, x0) / factorial(i))
        if i < order:
            current_expr = diff(current_expr, x)
    return coeffs


def ntaylor_coeffs(f: Callable,
                   x0: float,
                   order: int) -> list[mpnumeric]:
    return mpmath.taylor(f, x0, order)


def expr_from_coeffs(coeffs: Sequence[Number], var: Expr | Symbol) -> Expr:
    return Add(*[c*var**i for i, c in enumerate(coeffs)])


# Type checking doesn't like the signatures
# noinspection PyUnresolvedReferences
# noinspection PyTypeChecker
def _clear_denominators(l1: Sequence[Number], l2: Sequence[Number]) -> tuple[list[Number], list[Number]]:
    """Clear denominators from a list of SymPy numbers by multiplying by LCM of denominators."""
    lcm: Integer = lcm_list([r.q for r in l1 + l2 if isinstance(r, Rational)])
    return [n * lcm for n in l1], [n * lcm for n in l2]  # no inspection because crazy IDE type check bug


def simple_rationalize_float(x: float | mpnumeric | Number,
                             max_denominator: int = 10000,
                             tolerance: float = 1e-12) -> Rational | Float:
    """Convert float-like to simple rational, or keep as float if not clearly rational"""
    from fractions import Fraction
    try:
        float_val: float = float(x)
        # Use Fraction's built-in rational approximation
        frac: Fraction = Fraction(float_val).limit_denominator(max_denominator)

        # Only convert if it's a very close match
        if abs(float(frac) - float_val) < tolerance:
            return Rational(frac.numerator, frac.denominator)
        else:
            return Float(float_val)
    except:
        return Float(float(x))


# ====================== Numerical Pade (Fast, but potentially inaccurate implementation) ======================
def _pade_numerical(tc: list[mpnumeric],
                    m: int,
                    n: int) -> tuple[list[mpnumeric], list[mpnumeric]]:
    return mpmath.pade(tc, m, n)


# =========================== Verbose/Explicit Pade (slower symbolic implementation) ===========================
def _get_anti_diagonals(matrix: list[list]) -> list[list]:
    row_length: int = len(matrix)
    col_length: int = len(matrix[0])
    anti_diagonals: list[list] = [[] for _ in range(row_length + col_length - 1)]
    for c in range(col_length):
        for r in range(row_length):
            anti_diagonals[c + r].append(matrix[r][c])
    return anti_diagonals


def _expand_coeffs(v1: list[Expr | Number | float],
                   v2: list[Expr | Number | float]) -> list[Expr]:
    """Discrete Convolution"""
    m: list[list[Expr]] = [[e1 * e2 for e2 in v2] for e1 in v1]  # list matrix
    return [Add(*d) for d in _get_anti_diagonals(m)]


def _pade_verbose(tc: list[Number],
                  m: int,
                  n: int,
                  verbose: bool = False) -> tuple[tuple[Number, ...], tuple[Number, ...]]:
    unknown_a: list[Symbol] = [Symbol(f'a{i}') for i in range(m + 1)]
    unknown_b: list[Symbol] = [Symbol(f'b{i}') for i in range(1, n + 1)]
    if verbose:
        print('Step 1. Create rational function with numerator P and denominator Q, each with unknown coefficients:')
        _numerator = ' + '.join([f'a{i}' if i == 0 else
                                 f'a{i}*h' if i == 1 else
                                 f'a{i}*h^{i}' for i in range(m+1)])
        _denominator = ' + '.join([f'1' if i == 0 else
                                   f'b{i}*h' if i == 1 else
                                   f'b{i}*h^{i}' for i in range(n+1)])
        print(f'({_numerator})/({_denominator})')

        print('\nStep 2. Equate the rational function to the taylor series A so that the unknown coefficients may be solved:')
        _taylor_poly = ' + '.join([f'{c}' if i == 0 else
                                   f'{c}*h' if i == 1 else
                                   f'{c}*h^{i}' for i, c in enumerate(tc)])
        print(f'({_numerator})/({_denominator})', ' = ', _taylor_poly)

        print('\nStep 3. Multiply the rhs by the denominator of the lhs to get the equation in the form P = QA:')
        print(f'{_numerator}', ' = ', f'({_denominator}) ({_taylor_poly})')

        print('\nStep 4. Expand the RHS by performing discrete convolution on the coefficient vectors of Q and A (using a table):')
        print(f"\tQ's coeffs = {[1] + unknown_b}")
        print(f"\tA's coeffs = {tc}")
        _print_multiplication_table([1] + unknown_b, tc)

    convoluted: list[Expr] = _expand_coeffs(tc, [1] + unknown_b)[:m + n + 1]  # slice to get only the same number of equations as unknown variables.
    eqs: list[Expr] = [-a + c for a, c in zip_longest(unknown_a, convoluted, fillvalue=0)]
    if verbose:
        print(f'\nStep 5. Get the sum of the anti-diagonals from the above table to form the new coeffs (only as many terms as unknown coeffs we need to solve for, in this case {m+n+1}):')
        print(tabulate([[f'h^{i}', convoluted[i]] for i in range(len(convoluted))], headers=['Term', 'Coeff'], tablefmt='rounded_grid'))

        print('\nStep 6. Use these coefficients to setup a system of equations:')
        for rhs, lhs in zip_longest(unknown_a, convoluted, fillvalue=0):
            print(f'{' ' if rhs==0 else ''}{rhs} = {lhs}')

    sol: FiniteSet = linsolve(eqs, unknown_a + unknown_b)
    sol: Tuple = iter(sol).__next__()
    if verbose:
        print('\nStep 7. Solving the above system yields:')
        for v, s, in zip(unknown_a + unknown_b, sol):
            print(f'{v} = {s}')

        print('\nStep 8. Substituting these values back into the original rational function yields:')
        _a_sols = sol[:m + 1]
        _b_sols = (1, *sol[m + 1:])
        _numerator = ' + '.join([f'{c}' if i == 0 else
                                 f'{c}*h' if i == 1 else
                                 f'{c}*h^{i}' for i, c in enumerate(_a_sols)])
        _denominator = ' + '.join([f'{c}' if i == 0 else
                                 f'{c}*h' if i == 1 else
                                 f'{c}*h^{i}' for i, c in enumerate(_b_sols)])
        print(f'({_numerator})/({_denominator})')

        print('\nStep 9. `h` may be substituted for the original (x-c), and then expanded. `c` is the point at which both the Taylor series and the Padé approximation are centered at:')
        print(f'({_numerator})/({_denominator})'.replace('h', '(x-c)'))

    return sol[:m + 1], (1, *sol[m + 1:])  # returns known_a, known_b


# ================================ Pade (faster symbolic implementation) ================================
def _setup_pade_matrix(taylor_coeffs: list[Number], numerator_degree: int, denominator_degree: int) -> tuple[Matrix, Matrix]:
    """
    Set up the linear system for computing Padé approximants.

    A Padé approximant is a rational function P(h)/Q(h) that matches a given Taylor series
    to the highest possible order. This function constructs the matrix equation Ax = b
    where solving for x gives the coefficients of both the numerator P(h) and denominator Q(h).

    Mathematical Background:
    ------------------------
    We seek P(h) = a₀ + a₁h + ... + aₘhᵐ and Q(h) = 1 + b₁h + ... + bₙhⁿ such that:

        P(h) = Q(h) × (c₀ + c₁h + c₂h² + ...)

    Expanding the right side and equating coefficients of like powers of h gives us
    a system of linear equations in the unknown coefficients {a₀, a₁, ..., aₘ, b₁, ..., bₙ}.

    Parameters:
    -----------
    taylor_coeffs : array-like
        Taylor series coefficients [c₀, c₁, c₂, ...] where cᵢ is the coefficient of hⁱ.
        Must contain at least numerator_degree + denominator_degree + 1 coefficients.
    numerator_degree : int
        Maximum degree m of the numerator polynomial P(h).
    denominator_degree : int
        Maximum degree n of the denominator polynomial Q(h).

    Returns:
    --------
    A : sympy.Matrix
        Coefficient matrix of shape (m+n+1, m+n+1).
    b : sympy.Matrix
        Right-hand side vector of shape (m+n+1, 1).

    Notes:
    ------
    The solution vector x = [a₀, a₁, ..., aₘ, b₁, b₂, ..., bₙ] contains:
    - Numerator coefficients a₀ through aₘ (indices 0 to m)
    - Denominator coefficients b₁ through bₙ (indices m+1 to m+n)
    Note that b₀ = 1 is fixed by normalization and not included in the unknowns.
    """
    # Calculate total number of unknown coefficients: a₀...aₘ (m+1 terms) + b₁...bₙ (n terms)
    num_unknowns: int = numerator_degree + 1 + denominator_degree

    # Initialize coefficient matrix A and right-hand side vector b
    coefficient_matrix: Matrix = zeros(num_unknowns, num_unknowns)
    rhs_vector: Matrix = zeros(num_unknowns, 1)

    # Build one equation for each power of h (from h⁰ to h^(m+n))
    for power_of_h in range(num_unknowns):
        # Each equation enforces: [coeff of h^power_of_h in P(h)] = [coeff of h^power_of_h in Q(h)×taylor_series]

        # LEFT SIDE: Contribution from P(h) = a₀ + a₁h + ... + aₘhᵐ
        if power_of_h <= numerator_degree:
            # The coefficient of h^power_of_h in P(h) is exactly a_{power_of_h}
            coefficient_matrix[power_of_h, power_of_h] = 1

        # RIGHT SIDE: Contribution from Q(h) × taylor_series
        # When Q(h) = 1 + b₁h + ... + bₙhⁿ multiplies the Taylor series,
        # the coefficient of h^power_of_h comes from all term pairs where exponents sum to power_of_h

        # Loop over all terms in Q(h) that could contribute to h^power_of_h coefficient
        for q_term_power in range(min(power_of_h + 1, denominator_degree + 1)):
            # q_term_power is the exponent of the term in Q(h) we're considering
            # It ranges from 0 (constant term) to min(power_of_h, denominator_degree)

            # Find which Taylor coefficient we need: if Q term is h^q_term_power, and
            # we want total power power_of_h, then Taylor term must be h^(power_of_h - q_term_power)
            taylor_coeff_index: int = power_of_h - q_term_power

            # Check if the required Taylor coefficient exists in our input array
            if 0 <= taylor_coeff_index < len(taylor_coeffs):

                if q_term_power == 0:
                    # This is the constant term q₀ = 1 in Q(h)
                    # Contribution: 1 × c_{power_of_h} = c_{power_of_h} on the right side
                    rhs_vector[power_of_h] = taylor_coeffs[taylor_coeff_index]

                else:
                    # This is a variable term b_{q_term_power} × h^q_term_power in Q(h)
                    # Contribution: b_{q_term_power} × c_{taylor_coeff_index} on the right side
                    # Moving to left side gives: -b_{q_term_power} × c_{taylor_coeff_index}

                    # Find column index for b_{q_term_power} in our unknown vector
                    # Unknowns are ordered as [a₀, a₁, ..., aₘ, b₁, b₂, ..., bₙ]
                    # So b₁ is at index m+1, b₂ at m+2, etc.
                    # Therefore, b_{q_term_power} is at index m + q_term_power
                    denominator_coeff_column = numerator_degree + q_term_power

                    # Set the coefficient in the matrix (negative because moved to left side)
                    coefficient_matrix[power_of_h, denominator_coeff_column] = -taylor_coeffs[taylor_coeff_index]

    return coefficient_matrix, rhs_vector


def _pade(tc: list[Number],
          m: int,
          n: int) -> tuple[tuple[Number, ...], tuple[Number, ...]]:
    A: Matrix
    b: Matrix
    # Ax = b; we are solving for the vector x=[a0, a1, a2, ..., a_m, b1, b2, ..., b_n]
    A, b = _setup_pade_matrix(tc, m, n)
    sol: tuple[Number, ...] = tuple(A.solve(b))  # Matrix.solve() -> Matrix uses LU solver and is not optimal for Hankel structured matrices. Implementing specialized Hankel algorithms would provide the optimal time complexity.
    return sol[:m + 1], (1, *sol[m + 1:])


# ================================ Interface ================================
# noinspection PyUnboundLocalVariable
def pade(expr: Expr | Any,
         x: Symbol,
         x0: float,
         m: int,
         n: int,
         return_type: Literal['rational', 'pair', 'coeffs', 'dict'] = 'rational',
         expand_terms: bool = True,
         clear_coeff_denominators: bool = True,
         rationalize_mpmath_coeffs: bool = True,
         backend: Literal['auto', 'symbolic', 'verbose', 'mpmath'] = 'auto',
         _lambdified_expr: Callable = None) -> Expr | tuple[Sequence[Number] | Expr, Sequence[Number] | Expr] | dict:
    if backend == 'auto':
        backend = 'mpmath' if m + n > 20 else 'symbolic'

    # compute the pade approximation using `backend`
    a: Sequence[Number]
    b: Sequence[Number]
    if backend in ('symbolic', 'verbose'):
        # create taylor series
        tc: list[Number] = taylor_coeffs(expr, x, x0, m + n)
        # find pade approximation
        if backend == 'symbolic':
            a, b = _pade(tc, m, n)
        elif backend == 'verbose':
            a, b = _pade_verbose(tc, m, n, True)
    elif backend == 'mpmath':
        if _lambdified_expr is None:
            _lambdified_expr: Callable = lambdify(x, expr)
        # create taylor series
        tc: list[mpnumeric] = ntaylor_coeffs(_lambdified_expr, x0, m + n)
        # find pade approximation
        a, b = _pade_numerical(tc, m, n)
        if rationalize_mpmath_coeffs:
            a = [simple_rationalize_float(c) for c in a]
            b = [simple_rationalize_float(c) for c in b]
    else:
        raise ValueError(f"Backend '{backend}' not supported.")

    # clear
    if clear_coeff_denominators:
        a, b = _clear_denominators(a, b)

    # return route
    if return_type == 'coeffs':
        return a, b

    # create expression pair
    p: Expr = expr_from_coeffs(a, x - x0)
    q: Expr = expr_from_coeffs(b, x - x0)
    if expand_terms:
        p = expand_multinomial(p)
        q = expand_multinomial(q)

    # return route
    if return_type == 'pair':
        return p, q

    # create rational
    rf: Expr = p / q

    # return route
    if return_type == 'rational':
        return rf
    if return_type == 'dict':
        return {
            'rational': rf,
            'pair': (p, q),
            'coeffs': (a, b)
        }
    raise ValueError(f"Return type '{return_type}' not supported.")


# Test the function
if __name__ == "__main__":
    import sympy as sp
    from sympy import exp, sqrt
    from sympy.abc import x
    e = sqrt(x)
    r = pade(e, x, 1, 2, 2)
    print(r)
    # f = sympy.lambdify(x, e)
    # r = _pade_numerical(ntaylor_coeffs(f, 1, 4), 2, 2)
    # print(r)
    print(simple_rationalize_float(0.9782608695652174))
