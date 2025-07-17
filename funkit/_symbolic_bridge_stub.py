# Import required names here
from sympy import Expr

class SympyModule:
    def collect_const(self, *vars, Numbers=True):
        """A non-greedy collection of terms with similar number coefficients in
        an Add expr. If ``vars`` is given then only those constants will be
        targeted. Although any Number can also be targeted, if this is not
        desired set ``Numbers=False`` and no Float or Rational will be collected.
        
        Parameters
        ==========
        
        expr : SymPy expression
            This parameter defines the expression the expression from which
            terms with similar coefficients are to be collected. A non-Add
            expression is returned as it is.
        
        vars : variable length collection of Numbers, optional
            Specifies the constants to target for collection. Can be multiple in
            number.
        
        Numbers : bool
            Specifies to target all instance of
            :class:`sympy.core.numbers.Number` class. If ``Numbers=False``, then
            no Float or Rational will be collected.
        
        Returns
        =======
        
        expr : Expr
            Returns an expression with similar coefficient terms collected.
        """
    def collect_sqrt(self, evaluate=None):
        """Return expr with terms having common square roots collected together.
        If ``evaluate`` is False a count indicating the number of sqrt-containing
        terms will be returned and, if non-zero, the terms of the Add will be
        returned, else the expression itself will be returned as a single term.
        If ``evaluate`` is True, the expression with any collected terms will be
        returned.
        
        Note: since I = sqrt(-1), it is collected, too.
        """
    def convert_rationals_to_floats(self, n=15, exponent=False, dkeys=False):
        """Make all Rationals in expr Floats except those in exponents
        (unless the exponents flag is set to True) and those in undefined
        functions. When processing dictionaries, do not modify the keys
        unless ``dkeys=True``.
        """
    def expand_complex(self, deep=True):
        """Wrapper around expand that only uses the complex hint.  See the expand
        docstring for more information.
        """
    def expand_func(self, deep=True):
        """Wrapper around expand that only uses the func hint.  See the expand
        docstring for more information.
        """
    def expand_log(self, deep=True, force=False, factor=False):
        """Wrapper around expand that only uses the log hint.  See the expand
        docstring for more information.
        """
    def expand_mul(self, deep=True):
        """Wrapper around expand that only uses the mul hint.  See the expand
        docstring for more information.
        """
    def expand_multinomial(self, deep=True):
        """Wrapper around expand that only uses the multinomial hint.  See the expand
        docstring for more information.
        """
    def expand_power_base(self, deep=True, force=False):
        """Wrapper around expand that only uses the power_base hint.
        
        A wrapper to expand(power_base=True) which separates a power with a base
        that is a Mul into a product of powers, without performing any other
        expansions, provided that assumptions about the power's base and exponent
        allow.
        
        deep=False (default is True) will only apply to the top-level expression.
        
        force=True (default is False) will cause the expansion to ignore
        assumptions about the base and exponent. When False, the expansion will
        only happen if the base is non-negative or the exponent is an integer.
        """
    def expand_power_exp(self, deep=True):
        """Wrapper around expand that only uses the power_exp hint.
        
        See the expand docstring for more information.
        """
    def expand_trig(self, deep=True):
        """Wrapper around expand that only uses the trig hint.  See the expand
        docstring for more information.
        """
    def factor_nc(self):
        """Return the factored form of ``expr`` while handling non-commutative
        expressions.
        """
    def factor_terms(self, radical=False, clear=False, fraction=False, sign=True) -> Expr:
        """Remove common factors from terms in all arguments without
        changing the underlying structure of the expr. No expansion or
        simplification (and no processing of non-commutatives) is performed.
        
        Parameters
        ==========
        
        radical: bool, optional
            If radical=True then a radical common to all terms will be factored
            out of any Add sub-expressions of the expr.
        
        clear : bool, optional
            If clear=False (default) then coefficients will not be separated
            from a single Add if they can be distributed to leave one or more
            terms with integer coefficients.
        
        fraction : bool, optional
            If fraction=True (default is False) then a common denominator will be
            constructed for the expression.
        
        sign : bool, optional
            If sign=True (default) then even if the only factor in common is a -1,
            it will be factored out of the expression.
        """
    def gcd_terms(self, isprimitive=False, clear=True, fraction=True):
        """Compute the GCD of ``terms`` and put them together.
        
        Parameters
        ==========
        
        terms : Expr
            Can be an expression or a non-Basic sequence of expressions
            which will be handled as though they are terms from a sum.
        
        isprimitive : bool, optional
            If ``isprimitive`` is True the _gcd_terms will not run the primitive
            method on the terms.
        
        clear : bool, optional
            It controls the removal of integers from the denominator of an Add
            expression. When True (default), all numerical denominator will be cleared;
            when False the denominators will be cleared only if all terms had numerical
            denominators other than 1.
        
        fraction : bool, optional
            When True (default), will put the expression over a common
            denominator.
        """
    def horner(self, *gens, **args):
        """Rewrite a polynomial in Horner form.
        
        Among other applications, evaluation of a polynomial at a point is optimal
        when it is applied using the Horner scheme ([1]).
        """
    def hypersimp(self, k):
        """Given combinatorial term f(k) simplify its consecutive term ratio
        i.e. f(k+1)/f(k).  The input term can be composed of functions and
        integer sequences which have equivalent representation in terms
        of gamma special function.
        
        Explanation
        ===========
        
        The algorithm performs three basic steps:
        
        1. Rewrite all functions in terms of gamma, if possible.
        
        2. Rewrite all occurrences of gamma in terms of products
           of gamma and rising factorial with integer,  absolute
           constant exponent.
        
        3. Perform simplification of nested fractions, powers
           and if the resulting expression is a quotient of
           polynomials, reduce their total degree.
        
        If f(k) is hypergeometric then as result we arrive with a
        quotient of polynomials of minimal degree. Otherwise None
        is returned.
        
        For more information on the implemented algorithm refer to:
        
        1. W. Koepf, Algorithms for m-fold Hypergeometric Summation,
           Journal of Symbolic Computation (1995) 20, 399-417"""
    def is_sqrt(self):
        """Return True if expr is a sqrt, otherwise False."""
    def logcombine(self, force=False):
        """Takes logarithms and combines them using the following rules:
        
        - log(x) + log(y) == log(x*y) if both are positive
        - a*log(x) == log(x**a) if x is positive and a is real
        
        If ``force`` is ``True`` then the assumptions above will be assumed to hold if
        there is no assumption already in place on a quantity. For example, if
        ``a`` is imaginary or the argument negative, force will not perform a
        combination but if ``a`` is a symbol with no assumptions the change will
        take place.
        """
    def nfloat(self, n=15, exponent=False, dkeys=False):
        """Make all Rationals in expr Floats except those in exponents
        (unless the exponents flag is set to True) and those in undefined
        functions. When processing dictionaries, do not modify the keys
        unless ``dkeys=True``.
        """
    def nsolve(self, *args, dict=False, **kwargs):
        """Solve a nonlinear equation system numerically: ``nsolve(f, [args,] x0,
        modules=['mpmath'], **kwargs)``.
        
        Explanation
        ===========
        
        ``f`` is a vector function of symbolic expressions representing the system.
        *args* are the variables. If there is only one variable, this argument can
        be omitted. ``x0`` is a starting vector close to a solution.
        
        Use the modules keyword to specify which modules should be used to
        evaluate the function and the Jacobian matrix. Make sure to use a module
        that supports matrices. For more information on the syntax, please see the
        docstring of ``lambdify``.
        
        If the keyword arguments contain ``dict=True`` (default is False) ``nsolve``
        will return a list (perhaps empty) of solution mappings. This might be
        especially useful if you want to use ``nsolve`` as a fallback to solve since
        using the dict argument for both methods produces return values of
        consistent type structure. Please note: to keep this consistent with
        ``solve``, the solution will be returned in a list even though ``nsolve``
        (currently at least) only finds one solution at a time.
        
        Overdetermined systems are supported.
        """
    def nthroot(self, n, max_len=4, prec=15):
        """Compute a real nth-root of a sum of surds.
        
        Parameters
        ==========
        
        expr : sum of surds
        n : integer
        max_len : maximum number of surds passed as constants to ``nsimplify``
        
        Algorithm
        =========
        
        First ``nsimplify`` is used to get a candidate root; if it is not a
        root the minimal polynomial is computed; the answer is one of its
        roots.
        """
    def powdenest(self, force=False, polar=False):
        """Collect exponents on powers as assumptions allow.
        
        Explanation
        ===========
        
        Given ``(bb**be)**e``, this can be simplified as follows:
            * if ``bb`` is positive, or
            * ``e`` is an integer, or
            * ``|be| < 1`` then this simplifies to ``bb**(be*e)``
        
        Given a product of powers raised to a power, ``(bb1**be1 *
        bb2**be2...)**e``, simplification can be done as follows:
        
        - if e is positive, the gcd of all bei can be joined with e;
        - all non-negative bb can be separated from those that are negative
          and their gcd can be joined with e; autosimplification already
          handles this separation.
        - integer factors from powers that have integers in the denominator
          of the exponent can be removed from any term and the gcd of such
          integers can be joined with e
        
        Setting ``force`` to ``True`` will make symbols that are not explicitly
        negative behave as though they are positive, resulting in more
        denesting.
        
        Setting ``polar`` to ``True`` will do simplifications on the Riemann surface of
        the logarithm, also resulting in more denestings.
        
        When there are sums of logs in exp() then a product of powers may be
        obtained e.g. ``exp(3*(log(a) + 2*log(b)))`` - > ``a**3*b**6``.
        """
    def rad_rationalize(self, den):
        """Rationalize ``num/den`` by removing square roots in the denominator;
        num and den are sum of terms whose squares are positive rationals.
        """
    def rcollect(self, *vars):
        """Recursively collect sums in an expression.
        """
    def separatevars(self, symbols=[], dict=False, force=False):
        """Separates variables in an expression, if possible.  By
        default, it separates with respect to all symbols in an
        expression and collects constant coefficients that are
        independent of symbols.
        
        Explanation
        ===========
        
        If ``dict=True`` then the separated terms will be returned
        in a dictionary keyed to their corresponding symbols.
        By default, all symbols in the expression will appear as
        keys; if symbols are provided, then all those symbols will
        be used as keys, and any terms in the expression containing
        other symbols or non-symbols will be returned keyed to the
        string 'coeff'. (Passing None for symbols will return the
        expression in a dictionary keyed to 'coeff'.)
        
        If ``force=True``, then bases of powers will be separated regardless
        of assumptions on the symbols involved.
        
        Notes
        =====
        
        The order of the factors is determined by Mul, so that the
        separated expressions may not necessarily be grouped together.
        
        Although factoring is necessary to separate variables in some
        expressions, it is not necessary in all cases, so one should not
        count on the returned factors being factored.
        """
    def solve(self, *symbols, **flags):
        """Algebraically solves equations and systems of equations.
        
        Explanation
        ===========
        
        Currently supported:
            - polynomial
            - transcendental
            - piecewise combinations of the above
            - systems of linear and polynomial equations
            - systems containing relational expressions
            - systems implied by undetermined coefficients
        """
    def sqrt_depth(self) -> int:
        """Return the maximum depth of any square root argument of p.
        """
    def sqrtdenest(self, max_iter=3):
        """Denests sqrts in an expression that contain other square roots
        if possible, otherwise returns the expr unchanged. This is based on the
        algorithms of [1].
        """
    def symmetrize(self, *gens, **args):
        """Rewrite a polynomial in terms of elementary symmetric polynomials.
        
        A symmetric polynomial is a multivariate polynomial that remains invariant
        under any variable permutation, i.e., if `f = f(x_1, x_2, \dots, x_n)`,
        then `f = f(x_{i_1}, x_{i_2}, \dots, x_{i_n})`, where
        `(i_1, i_2, \dots, i_n)` is a permutation of `(1, 2, \dots, n)` (an
        element of the group `S_n`).
        
        Returns a tuple of symmetric polynomials ``(f1, f2, ..., fn)`` such that
        ``f = f1 + f2 + ... + fn``.
        """
