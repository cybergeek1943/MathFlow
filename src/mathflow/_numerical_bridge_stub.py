# Import required names here
import numpy as np
import numpy
from numpy import fmin
from math import inf
from typing import Union, Iterable, Optional, Callable
from scipy.optimize import Bounds, OptimizeResult

# noinspection PyUnresolvedReferences,HttpUrlsUsage,PyIncorrectDocstring,PyPep8Naming,PyShadowingBuiltins,PyDefaultArgument
class ScipyModule:
    def basinhopping(self, x0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs=None, take_step=None, accept_test=None, callback=None, interval=50, disp=False, niter_success=None, rng=None, target_accept_rate=0.5, stepwise_factor=0.9):
        r"""    
        
        
        Find the global minimum of a function using the basin-hopping algorithm.
        
        Basin-hopping is a two-phase method that combines a global stepping
        algorithm with local minimization at each step. Designed to mimic
        the natural process of energy minimization of clusters of atoms, it works
        well for similar problems with "funnel-like, but rugged" energy landscapes
        [5]_.
        
        As the step-taking, step acceptance, and minimization methods are all
        customizable, this function can also be used to implement other two-phase
        methods.
        
        Parameters
        ----------
        func : callable ``f(x, *args)``
            Function to be optimized.  ``args`` can be passed as an optional item
            in the dict `minimizer_kwargs`
        x0 : array_like
            Initial guess.
        niter : integer, optional
            The number of basin-hopping iterations. There will be a total of
            ``niter + 1`` runs of the local minimizer.
        T : float, optional
            The "temperature" parameter for the acceptance or rejection criterion.
            Higher "temperatures" mean that larger jumps in function value will be
            accepted.  For best results `T` should be comparable to the
            separation (in function value) between local minima.
        stepsize : float, optional
            Maximum step size for use in the random displacement.
        minimizer_kwargs : dict, optional
            Extra keyword arguments to be passed to the local minimizer
            `scipy.optimize.minimize` Some important options could be:
            
            method : str
                The minimization method (e.g. ``"L-BFGS-B"``)
            args : tuple
                Extra arguments passed to the objective function (`func`) and
                its derivatives (Jacobian, Hessian).
        take_step : callable ``take_step(x)``, optional
            Replace the default step-taking routine with this routine. The default
            step-taking routine is a random displacement of the coordinates, but
            other step-taking algorithms may be better for some systems.
            `take_step` can optionally have the attribute ``take_step.stepsize``.
            If this attribute exists, then `basinhopping` will adjust
            ``take_step.stepsize`` in order to try to optimize the global minimum
            search.
        accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
            Define a test which will be used to judge whether to accept the
            step. This will be used in addition to the Metropolis test based on
            "temperature" `T`. The acceptable return values are True,
            False, or ``"force accept"``. If any of the tests return False
            then the step is rejected. If the latter, then this will override any
            other tests in order to accept the step. This can be used, for example,
            to forcefully escape from a local minimum that `basinhopping` is
            trapped in.
        callback : callable, ``callback(x, f, accept)``, optional
            A callback function which will be called for all minima found. ``x``
            and ``f`` are the coordinates and function value of the trial minimum,
            and ``accept`` is whether that minimum was accepted. This can
            be used, for example, to save the lowest N minima found. Also,
            `callback` can be used to specify a user defined stop criterion by
            optionally returning True to stop the `basinhopping` routine.
        interval : integer, optional
            interval for how often to update the `stepsize`
        disp : bool, optional
            Set to True to print status messages
        niter_success : integer, optional
            Stop the run if the global minimum candidate remains the same for this
            number of iterations.
        rng : {None, int, `numpy.random.Generator`}, optional
            If `rng` is passed by keyword, types other than `numpy.random.Generator` are
            passed to `numpy.random.default_rng` to instantiate a ``Generator``.
            If `rng` is already a ``Generator`` instance, then the provided instance is
            used. Specify `rng` for repeatable function behavior.
        
            If this argument is passed by position or `seed` is passed by keyword,
            legacy behavior for the argument `seed` applies:
        
            - If `seed` is None (or `numpy.random`), the `numpy.random.RandomState`
              singleton is used.
            - If `seed` is an int, a new ``RandomState`` instance is used,
              seeded with `seed`.
            - If `seed` is already a ``Generator`` or ``RandomState`` instance then
              that instance is used.
        
            .. versionchanged:: 1.15.0
                As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
                transition from use of `numpy.random.RandomState` to
                `numpy.random.Generator`, this keyword was changed from `seed` to `rng`.
                For an interim period, both keywords will continue to work, although only one
                may be specified at a time. After the interim period, function calls using the
                `seed` keyword will emit warnings. The behavior of both `seed` and
                `rng` are outlined above, but only the `rng` keyword should be used in new code.
                
            The random numbers generated only affect the default Metropolis
            `accept_test` and the default `take_step`. If you supply your own
            `take_step` and `accept_test`, and these functions use random
            number generation, then those functions are responsible for the state
            of their random number generator.
        target_accept_rate : float, optional
            The target acceptance rate that is used to adjust the `stepsize`.
            If the current acceptance rate is greater than the target,
            then the `stepsize` is increased. Otherwise, it is decreased.
            Range is (0, 1). Default is 0.5.
            
            .. versionadded:: 1.8.0
        stepwise_factor : float, optional
            The `stepsize` is multiplied or divided by this stepwise factor upon
            each update. Range is (0, 1). Default is 0.9.
            
            .. versionadded:: 1.8.0
        
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            Important attributes are: ``x`` the solution array, ``fun`` the value
            of the function at the solution, and ``message`` which describes the
            cause of the termination. The ``OptimizeResult`` object returned by the
            selected minimizer at the lowest minimum is also contained within this
            object and can be accessed through the ``lowest_optimization_result``
            attribute.  See `OptimizeResult` for a description of other attributes.
        
        See Also
        --------
        
        :func:`minimize`
            The local minimization function called once for each basinhopping step. `minimizer_kwargs` is passed to this routine.
        
        
        Notes
        -----
        Basin-hopping is a stochastic algorithm which attempts to find the global
        minimum of a smooth scalar function of one or more variables [1]_ [2]_ [3]_
        [4]_. The algorithm in its current form was described by David Wales and
        Jonathan Doye [2]_ http://www-wales.ch.cam.ac.uk/.
        
        The algorithm is iterative with each cycle composed of the following
        features
        
        1) random perturbation of the coordinates
        
        2) local minimization
        
        3) accept or reject the new coordinates based on the minimized function
           value
        
        The acceptance test used here is the Metropolis criterion of standard Monte
        Carlo algorithms, although there are many other possibilities [3]_.
        
        This global minimization method has been shown to be extremely efficient
        for a wide variety of problems in physics and chemistry. It is
        particularly useful when the function has many minima separated by large
        barriers. See the `Cambridge Cluster Database
        <https://www-wales.ch.cam.ac.uk/CCD.html>`_ for databases of molecular
        systems that have been optimized primarily using basin-hopping. This
        database includes minimization problems exceeding 300 degrees of freedom.
        
        See the free software program `GMIN <https://www-wales.ch.cam.ac.uk/GMIN>`_
        for a Fortran implementation of basin-hopping. This implementation has many
        variations of the procedure described above, including more
        advanced step taking algorithms and alternate acceptance criterion.
        
        For stochastic global optimization there is no way to determine if the true
        global minimum has actually been found. Instead, as a consistency check,
        the algorithm can be run from a number of different random starting points
        to ensure the lowest minimum found in each example has converged to the
        global minimum. For this reason, `basinhopping` will by default simply
        run for the number of iterations `niter` and return the lowest minimum
        found. It is left to the user to ensure that this is in fact the global
        minimum.
        
        Choosing `stepsize`:  This is a crucial parameter in `basinhopping` and
        depends on the problem being solved. The step is chosen uniformly in the
        region from x0-stepsize to x0+stepsize, in each dimension. Ideally, it
        should be comparable to the typical separation (in argument values) between
        local minima of the function being optimized. `basinhopping` will, by
        default, adjust `stepsize` to find an optimal value, but this may take
        many iterations. You will get quicker results if you set a sensible
        initial value for ``stepsize``.
        
        Choosing `T`: The parameter `T` is the "temperature" used in the
        Metropolis criterion. Basinhopping steps are always accepted if
        ``func(xnew) < func(xold)``. Otherwise, they are accepted with
        probability::
        
            exp( -(func(xnew) - func(xold)) / T )
        
        So, for best results, `T` should to be comparable to the typical
        difference (in function values) between local minima. (The height of
        "walls" between local minima is irrelevant.)
        
        If `T` is 0, the algorithm becomes Monotonic Basin-Hopping, in which all
        steps that increase energy are rejected.
        
        .. versionadded:: 0.12.0
        """
    def bisect(self, a, b, args=(), xtol=2e-12, rtol=np.float64(8.881784197001252e-16), maxiter=100, full_output=False, disp=True):
        r"""
        Find root of a function within an interval using bisection.
        
        Basic bisection routine to find a root of the function `f` between the
        arguments `a` and `b`. `f(a)` and `f(b)` cannot have the same signs.
        Slow but sure.
        
        Parameters
        ----------
        f : function
            Python function returning a number.  `f` must be continuous, and
            f(a) and f(b) must have opposite signs.
        a : scalar
            One end of the bracketing interval [a,b].
        b : scalar
            The other end of the bracketing interval [a,b].
        xtol : number, optional
            The computed root ``x0`` will satisfy ``np.allclose(x, x0,
            atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
            parameter must be positive.
        rtol : number, optional
            The computed root ``x0`` will satisfy ``np.allclose(x, x0,
            atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
            parameter cannot be smaller than its default value of
            ``4*np.finfo(float).eps``.
        maxiter : int, optional
            If convergence is not achieved in `maxiter` iterations, an error is
            raised. Must be >= 0.
        args : tuple, optional
            Containing extra arguments for the function `f`.
            `f` is called by ``apply(f, (x)+args)``.
        full_output : bool, optional
            If `full_output` is False, the root is returned. If `full_output` is
            True, the return value is ``(x, r)``, where x is the root, and r is
            a `RootResults` object.
        disp : bool, optional
            If True, raise RuntimeError if the algorithm didn't converge.
            Otherwise, the convergence status is recorded in a `RootResults`
            return object.
        
        Returns
        -------
        root : float
            Root of `f` between `a` and `b`.
        r : `RootResults` (present if ``full_output = True``)
            Object containing information about the convergence. In particular,
            ``r.converged`` is True if the routine converged.
        """
    def bracket_minimum(self, xm0, xl0=None, xr0=None, xmin=None, xmax=None, factor=None, args=(), maxiter=1000):
        r"""Bracket the minimum of a unimodal, real-valued function of a real variable.
        
        For each element of the output of `f`, `bracket_minimum` seeks the scalar
        bracket points ``xl < xm < xr`` such that ``fl >= fm <= fr`` where one of the
        inequalities is strict.
        
        The function is guaranteed to find a valid bracket if the function is
        strongly unimodal, but it may find a bracket under other conditions.
        
        This function works elementwise when `xm0`, `xl0`, `xr0`, `xmin`, `xmax`, `factor`,
        and the elements of `args` are (mutually broadcastable) arrays.
        
        Parameters
        ----------
        f : callable
            The function for which the root is to be bracketed. The signature must be::
        
                f(x: array, *args) -> array
        
            where each element of ``x`` is a finite real and ``args`` is a tuple,
            which may contain an arbitrary number of arrays that are broadcastable
            with ``x``.
        
            `f` must be an elementwise function: each element ``f(x)[i]``
            must equal ``f(x[i])`` for all indices ``i``. It must not mutate the
            array ``x`` or the arrays in ``args``.
        xm0: float array_like
            Starting guess for middle point of bracket.
        xl0, xr0: float array_like, optional
            Starting guesses for left and right endpoints of the bracket. Must
            be broadcastable with all other array inputs.
        xmin, xmax : float array_like, optional
            Minimum and maximum allowable endpoints of the bracket, inclusive. Must
            be broadcastable with all other array inputs.
        factor : float array_like, default: 2
            The factor used to grow the bracket. See Notes.
        args : tuple of array_like, optional
            Additional positional array arguments to be passed to `f`.
            If the callable for which the root is desired requires arguments that are
            not broadcastable with `x`, wrap that callable with `f` such that `f`
            accepts only `x` and broadcastable ``*args``.
        maxiter : int, default: 1000
            The maximum number of iterations of the algorithm to perform.
        
        Returns
        -------
        res : _RichResult
            An object similar to an instance of `scipy.optimize.OptimizeResult` with the
            following attributes. The descriptions are written as though the values will
            be scalars; however, if `f` returns an array, the outputs will be
            arrays of the same shape.
        
            success : bool array
                ``True`` where the algorithm terminated successfully (status ``0``);
                ``False`` otherwise.
            status : int array
                An integer representing the exit status of the algorithm.
        
                - ``0`` : The algorithm produced a valid bracket.
                - ``-1`` : The bracket expanded to the allowable limits. Assuming
                  unimodality, this implies the endpoint at the limit is a minimizer.
                - ``-2`` : The maximum number of iterations was reached.
                - ``-3`` : A non-finite value was encountered.
                - ``-4`` : ``None`` shall pass.
                - ``-5`` : The initial bracket does not satisfy
                  `xmin <= xl0 < xm0 < xr0 <= xmax`.
        
            bracket : 3-tuple of float arrays
                The left, middle, and right points of the bracket, if the algorithm
                terminated successfully.
            f_bracket : 3-tuple of float arrays
                The function value at the left, middle, and right points of the bracket.
            nfev : int array
                The number of abscissae at which `f` was evaluated to find the root.
                This is distinct from the number of times `f` is *called* because the
                the function may evaluated at multiple points in a single call.
            nit : int array
                The number of iterations of the algorithm that were performed.
        
        Notes
        -----
        Similar to `scipy.optimize.bracket`, this function seeks to find real
        points ``xl < xm < xr`` such that ``f(xl) >= f(xm)`` and ``f(xr) >= f(xm)``,
        where at least one of the inequalities is strict. Unlike `scipy.optimize.bracket`,
        this function can operate in a vectorized manner on array input, so long as
        the input arrays are broadcastable with each other. Also unlike
        `scipy.optimize.bracket`, users may specify minimum and maximum endpoints
        for the desired bracket.
        
        Given an initial trio of points ``xl = xl0``, ``xm = xm0``, ``xr = xr0``,
        the algorithm checks if these points already give a valid bracket. If not,
        a new endpoint, ``w`` is chosen in the "downhill" direction, ``xm`` becomes the new
        opposite endpoint, and either `xl` or `xr` becomes the new middle point,
        depending on which direction is downhill. The algorithm repeats from here.
        
        The new endpoint `w` is chosen differently depending on whether or not a
        boundary `xmin` or `xmax` has been set in the downhill direction. Without
        loss of generality, suppose the downhill direction is to the right, so that
        ``f(xl) > f(xm) > f(xr)``. If there is no boundary to the right, then `w`
        is chosen to be ``xr + factor * (xr - xm)`` where `factor` is controlled by
        the user (defaults to 2.0) so that step sizes increase in geometric proportion.
        If there is a boundary, `xmax` in this case, then `w` is chosen to be
        ``xmax - (xmax - xr)/factor``, with steps slowing to a stop at
        `xmax`. This cautious approach ensures that a minimum near but distinct from
        the boundary isn't missed while also detecting whether or not the `xmax` is
        a minimizer when `xmax` is reached after a finite number of steps.
        
        See Also
        --------
        scipy.optimize.bracket
        scipy.optimize.elementwise.find_minimum
        """
    def bracket_root(self, xl0, xr0=None, xmin=None, xmax=None, factor=None, args=(), maxiter=1000):
        r"""Bracket the root of a monotonic, real-valued function of a real variable.
        
        For each element of the output of `f`, `bracket_root` seeks the scalar
        bracket endpoints ``xl`` and ``xr`` such that ``sign(f(xl)) == -sign(f(xr))``
        elementwise.
        
        The function is guaranteed to find a valid bracket if the function is monotonic,
        but it may find a bracket under other conditions.
        
        This function works elementwise when `xl0`, `xr0`, `xmin`, `xmax`, `factor`, and
        the elements of `args` are (mutually broadcastable) arrays.
        
        Parameters
        ----------
        f : callable
            The function for which the root is to be bracketed. The signature must be::
        
                f(x: array, *args) -> array
        
            where each element of ``x`` is a finite real and ``args`` is a tuple,
            which may contain an arbitrary number of arrays that are broadcastable
            with ``x``.
        
            `f` must be an elementwise function: each element ``f(x)[i]``
            must equal ``f(x[i])`` for all indices ``i``. It must not mutate the
            array ``x`` or the arrays in ``args``.
        xl0, xr0: float array_like
            Starting guess of bracket, which need not contain a root. If `xr0` is
            not provided, ``xr0 = xl0 + 1``. Must be broadcastable with all other
            array inputs.
        xmin, xmax : float array_like, optional
            Minimum and maximum allowable endpoints of the bracket, inclusive. Must
            be broadcastable with all other array inputs.
        factor : float array_like, default: 2
            The factor used to grow the bracket. See Notes.
        args : tuple of array_like, optional
            Additional positional array arguments to be passed to `f`.
            If the callable for which the root is desired requires arguments that are
            not broadcastable with `x`, wrap that callable with `f` such that `f`
            accepts only `x` and broadcastable ``*args``.
        maxiter : int, default: 1000
            The maximum number of iterations of the algorithm to perform.
        
        Returns
        -------
        res : _RichResult
            An object similar to an instance of `scipy.optimize.OptimizeResult` with the
            following attributes. The descriptions are written as though the values will
            be scalars; however, if `f` returns an array, the outputs will be
            arrays of the same shape.
        
            success : bool array
                ``True`` where the algorithm terminated successfully (status ``0``);
                ``False`` otherwise.
            status : int array
                An integer representing the exit status of the algorithm.
        
                - ``0`` : The algorithm produced a valid bracket.
                - ``-1`` : The bracket expanded to the allowable limits without success.
                - ``-2`` : The maximum number of iterations was reached.
                - ``-3`` : A non-finite value was encountered.
                - ``-4`` : Iteration was terminated by `callback`.
                - ``-5``: The initial bracket does not satisfy`xmin <= xl0 < xr0 < xmax`.
                
            bracket : 2-tuple of float arrays
                The lower and upper endpoints of the bracket, if the algorithm
                terminated successfully.
            f_bracket : 2-tuple of float arrays
                The values of `f` evaluated at the endpoints of ``res.bracket``,
                respectively.
            nfev : int array
                The number of abscissae at which `f` was evaluated to find the root.
                This is distinct from the number of times `f` is *called* because the
                the function may evaluated at multiple points in a single call.
            nit : int array
                The number of iterations of the algorithm that were performed.
        
        Notes
        -----
        This function generalizes an algorithm found in pieces throughout the
        `scipy.stats` codebase. The strategy is to iteratively grow the bracket `(l, r)`
        until ``f(l) < 0 < f(r)`` or ``f(r) < 0 < f(l)``. The bracket grows to the left
        as follows.
        
        - If `xmin` is not provided, the distance between `xl0` and `l` is iteratively
          increased by `factor`.
        - If `xmin` is provided, the distance between `xmin` and `l` is iteratively
          decreased by `factor`. Note that this also *increases* the bracket size.
        
        Growth of the bracket to the right is analogous.
        
        Growth of the bracket in one direction stops when the endpoint is no longer
        finite, the function value at the endpoint is no longer finite, or the
        endpoint reaches its limiting value (`xmin` or `xmax`). Iteration terminates
        when the bracket stops growing in both directions, the bracket surrounds
        the root, or a root is found (by chance).
        
        If two brackets are found - that is, a bracket is found on both sides in
        the same iteration, the smaller of the two is returned.
        
        If roots of the function are found, both `xl` and `xr` are set to the
        leftmost root.
        
        See Also
        --------
        find_root
        """
    def brenth(self, a, b, args=(), xtol=2e-12, rtol=np.float64(8.881784197001252e-16), maxiter=100, full_output=False, disp=True):
        r"""Find a root of a function in a bracketing interval using Brent's
        method with hyperbolic extrapolation.
        
        A variation on the classic Brent routine to find a root of the function f
        between the arguments a and b that uses hyperbolic extrapolation instead of
        inverse quadratic extrapolation. Bus & Dekker (1975) guarantee convergence
        for this method, claiming that the upper bound of function evaluations here
        is 4 or 5 times that of bisection.
        f(a) and f(b) cannot have the same signs. Generally, on a par with the
        brent routine, but not as heavily tested. It is a safe version of the
        secant method that uses hyperbolic extrapolation.
        The version here is by Chuck Harris, and implements Algorithm M of
        [BusAndDekker1975]_, where further details (convergence properties,
        additional remarks and such) can be found
        
        Parameters
        ----------
        f : function
            Python function returning a number. f must be continuous, and f(a) and
            f(b) must have opposite signs.
        a : scalar
            One end of the bracketing interval [a,b].
        b : scalar
            The other end of the bracketing interval [a,b].
        xtol : number, optional
            The computed root ``x0`` will satisfy ``np.allclose(x, x0,
            atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
            parameter must be positive. As with `brentq`, for nice
            functions the method will often satisfy the above condition
            with ``xtol/2`` and ``rtol/2``.
        rtol : number, optional
            The computed root ``x0`` will satisfy ``np.allclose(x, x0,
            atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
            parameter cannot be smaller than its default value of
            ``4*np.finfo(float).eps``. As with `brentq`, for nice functions
            the method will often satisfy the above condition with
            ``xtol/2`` and ``rtol/2``.
        maxiter : int, optional
            If convergence is not achieved in `maxiter` iterations, an error is
            raised. Must be >= 0.
        args : tuple, optional
            Containing extra arguments for the function `f`.
            `f` is called by ``apply(f, (x)+args)``.
        full_output : bool, optional
            If `full_output` is False, the root is returned. If `full_output` is
            True, the return value is ``(x, r)``, where `x` is the root, and `r` is
            a `RootResults` object.
        disp : bool, optional
            If True, raise RuntimeError if the algorithm didn't converge.
            Otherwise, the convergence status is recorded in any `RootResults`
            return object.
        
        Returns
        -------
        root : float
            Root of `f` between `a` and `b`.
        r : `RootResults` (present if ``full_output = True``)
            Object containing information about the convergence. In particular,
            ``r.converged`` is True if the routine converged.
        
        See Also
        --------
        fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg : multivariate local optimizers
        leastsq : nonlinear least squares minimizer
        fmin_l_bfgs_b, fmin_tnc, fmin_cobyla : constrained multivariate optimizers
        basinhopping, differential_evolution, brute : global optimizers
        fminbound, brent, golden, bracket : local scalar minimizers
        fsolve : N-D root-finding
        brentq, ridder, bisect, newton : 1-D root-finding
        fixed_point : scalar fixed-point finder
        """
    def brentq(self, a, b, args=(), xtol=2e-12, rtol=np.float64(8.881784197001252e-16), maxiter=100, full_output=False, disp=True):
        r"""
        Find a root of a function in a bracketing interval using Brent's method.
        
        Uses the classic Brent's method to find a root of the function `f` on
        the sign changing interval [a , b]. Generally considered the best of the
        rootfinding routines here. It is a safe version of the secant method that
        uses inverse quadratic extrapolation. Brent's method combines root
        bracketing, interval bisection, and inverse quadratic interpolation. It is
        sometimes known as the van Wijngaarden-Dekker-Brent method. Brent (1973)
        claims convergence is guaranteed for functions computable within [a,b].
        
        [Brent1973]_ provides the classic description of the algorithm. Another
        description can be found in a recent edition of Numerical Recipes, including
        [PressEtal1992]_. A third description is at
        http://mathworld.wolfram.com/BrentsMethod.html. It should be easy to
        understand the algorithm just by reading our code. Our code diverges a bit
        from standard presentations: we choose a different formula for the
        extrapolation step.
        
        Parameters
        ----------
        f : function
            Python function returning a number. The function :math:`f`
            must be continuous, and :math:`f(a)` and :math:`f(b)` must
            have opposite signs.
        a : scalar
            One end of the bracketing interval :math:`[a, b]`.
        b : scalar
            The other end of the bracketing interval :math:`[a, b]`.
        xtol : number, optional
            The computed root ``x0`` will satisfy ``np.allclose(x, x0,
            atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
            parameter must be positive. For nice functions, Brent's
            method will often satisfy the above condition with ``xtol/2``
            and ``rtol/2``. [Brent1973]_
        rtol : number, optional
            The computed root ``x0`` will satisfy ``np.allclose(x, x0,
            atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
            parameter cannot be smaller than its default value of
            ``4*np.finfo(float).eps``. For nice functions, Brent's
            method will often satisfy the above condition with ``xtol/2``
            and ``rtol/2``. [Brent1973]_
        maxiter : int, optional
            If convergence is not achieved in `maxiter` iterations, an error is
            raised. Must be >= 0.
        args : tuple, optional
            Containing extra arguments for the function `f`.
            `f` is called by ``apply(f, (x)+args)``.
        full_output : bool, optional
            If `full_output` is False, the root is returned. If `full_output` is
            True, the return value is ``(x, r)``, where `x` is the root, and `r` is
            a `RootResults` object.
        disp : bool, optional
            If True, raise RuntimeError if the algorithm didn't converge.
            Otherwise, the convergence status is recorded in any `RootResults`
            return object.
        
        Returns
        -------
        root : float
            Root of `f` between `a` and `b`.
        r : `RootResults` (present if ``full_output = True``)
            Object containing information about the convergence. In particular,
            ``r.converged`` is True if the routine converged.
        
        See Also
        --------
        fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg : multivariate local optimizers
        leastsq : nonlinear least squares minimizer
        fmin_l_bfgs_b, fmin_tnc, fmin_cobyla : constrained multivariate optimizers
        basinhopping, differential_evolution, brute : global optimizers
        fminbound, brent, golden, bracket : local scalar minimizers
        fsolve : N-D root-finding
        brenth, ridder, bisect, newton : 1-D root-finding
        fixed_point : scalar fixed-point finder
        
        Notes
        -----
        `f` must be continuous.  f(a) and f(b) must have opposite signs.
        """
    def brute(self, ranges, args=(), Ns=20, full_output=0, finish=fmin, disp=False, workers=1):
        r"""Minimize a function over a given range by brute force.
        
        Uses the "brute force" method, i.e., computes the function's value
        at each point of a multidimensional grid of points, to find the global
        minimum of the function.
        
        The function is evaluated everywhere in the range with the datatype of the
        first call to the function, as enforced by the ``vectorize`` NumPy
        function. The value and type of the function evaluation returned when
        ``full_output=True`` are affected in addition by the ``finish`` argument
        (see Notes).
        
        The brute force approach is inefficient because the number of grid points
        increases exponentially - the number of grid points to evaluate is
        ``Ns ** len(x)``. Consequently, even with coarse grid spacing, even
        moderately sized problems can take a long time to run, and/or run into
        memory limitations.
        
        Parameters
        ----------
        func : callable
            The objective function to be minimized. Must be in the
            form ``f(x, *args)``, where ``x`` is the argument in
            the form of a 1-D array and ``args`` is a tuple of any
            additional fixed parameters needed to completely specify
            the function.
        ranges : tuple
            Each component of the `ranges` tuple must be either a
            "slice object" or a range tuple of the form ``(low, high)``.
            The program uses these to create the grid of points on which
            the objective function will be computed. See `Note 2` for
            more detail.
        args : tuple, optional
            Any additional fixed parameters needed to completely specify
            the function.
        Ns : int, optional
            Number of grid points along the axes, if not otherwise
            specified. See `Note2`.
        full_output : bool, optional
            If True, return the evaluation grid and the objective function's
            values on it.
        finish : callable, optional
            An optimization function that is called with the result of brute force
            minimization as initial guess. `finish` should take `func` and
            the initial guess as positional arguments, and take `args` as
            keyword arguments. It may additionally take `full_output`
            and/or `disp` as keyword arguments. Use None if no "polishing"
            function is to be used. See Notes for more details.
        disp : bool, optional
            Set to True to print convergence messages from the `finish` callable.
        workers : int or map-like callable, optional
            If `workers` is an int the grid is subdivided into `workers`
            sections and evaluated in parallel (uses
            `multiprocessing.Pool <multiprocessing>`).
            Supply `-1` to use all cores available to the Process.
            Alternatively supply a map-like callable, such as
            `multiprocessing.Pool.map` for evaluating the grid in parallel.
            This evaluation is carried out as ``workers(func, iterable)``.
            Requires that `func` be pickleable.
        
            .. versionadded:: 1.3.0
        
        Returns
        -------
        x0 : ndarray
            A 1-D array containing the coordinates of a point at which the
            objective function had its minimum value. (See `Note 1` for
            which point is returned.)
        fval : float
            Function value at the point `x0`. (Returned when `full_output` is
            True.)
        grid : tuple
            Representation of the evaluation grid. It has the same
            length as `x0`. (Returned when `full_output` is True.)
        Jout : ndarray
            Function values at each point of the evaluation
            grid, i.e., ``Jout = func(*grid)``. (Returned
            when `full_output` is True.)
        
        See Also
        --------
        basinhopping, differential_evolution
        
        Notes
        -----
        *Note 1*: The program finds the gridpoint at which the lowest value
        of the objective function occurs. If `finish` is None, that is the
        point returned. When the global minimum occurs within (or not very far
        outside) the grid's boundaries, and the grid is fine enough, that
        point will be in the neighborhood of the global minimum.
        
        However, users often employ some other optimization program to
        "polish" the gridpoint values, i.e., to seek a more precise
        (local) minimum near `brute's` best gridpoint.
        The `brute` function's `finish` option provides a convenient way to do
        that. Any polishing program used must take `brute's` output as its
        initial guess as a positional argument, and take `brute's` input values
        for `args` as keyword arguments, otherwise an error will be raised.
        It may additionally take `full_output` and/or `disp` as keyword arguments.
        
        `brute` assumes that the `finish` function returns either an
        `OptimizeResult` object or a tuple in the form:
        ``(xmin, Jmin, ... , statuscode)``, where ``xmin`` is the minimizing
        value of the argument, ``Jmin`` is the minimum value of the objective
        function, "..." may be some other returned values (which are not used
        by `brute`), and ``statuscode`` is the status code of the `finish` program.
        
        Note that when `finish` is not None, the values returned are those
        of the `finish` program, *not* the gridpoint ones. Consequently,
        while `brute` confines its search to the input grid points,
        the `finish` program's results usually will not coincide with any
        gridpoint, and may fall outside the grid's boundary. Thus, if a
        minimum only needs to be found over the provided grid points, make
        sure to pass in ``finish=None``.
        
        *Note 2*: The grid of points is a `numpy.mgrid` object.
        For `brute` the `ranges` and `Ns` inputs have the following effect.
        Each component of the `ranges` tuple can be either a slice object or a
        two-tuple giving a range of values, such as (0, 5). If the component is a
        slice object, `brute` uses it directly. If the component is a two-tuple
        range, `brute` internally converts it to a slice object that interpolates
        `Ns` points from its low-value to its high-value, inclusive.
        """
    def cubature(self, a, b, rule='gk21', rtol=1e-08, atol=0, max_subdivisions=10000, args=(), workers=1, points=None):
        r"""
        Adaptive cubature of multidimensional array-valued function.
        
        Given an arbitrary integration rule, this function returns an estimate of the
        integral to the requested tolerance over the region defined by the arrays `a` and
        `b` specifying the corners of a hypercube.
        
        Convergence is not guaranteed for all integrals.
        
        Parameters
        ----------
        f : callable
            Function to integrate. `f` must have the signature::
        
                f(x : ndarray, *args) -> ndarray
        
            `f` should accept arrays ``x`` of shape::
        
                (npoints, ndim)
        
            and output arrays of shape::
        
                (npoints, output_dim_1, ..., output_dim_n)
        
            In this case, `cubature` will return arrays of shape::
        
                (output_dim_1, ..., output_dim_n)
        a, b : array_like
            Lower and upper limits of integration as 1D arrays specifying the left and right
            endpoints of the intervals being integrated over. Limits can be infinite.
        rule : str, optional
            Rule used to estimate the integral. If passing a string, the options are
            "gauss-kronrod" (21 node), or "genz-malik" (degree 7). If a rule like
            "gauss-kronrod" is specified for an ``n``-dim integrand, the corresponding
            Cartesian product rule is used. "gk21", "gk15" are also supported for
            compatibility with `quad_vec`. See Notes.
        rtol, atol : float, optional
            Relative and absolute tolerances. Iterations are performed until the error is
            estimated to be less than ``atol + rtol * abs(est)``. Here `rtol` controls
            relative accuracy (number of correct digits), while `atol` controls absolute
            accuracy (number of correct decimal places). To achieve the desired `rtol`, set
            `atol` to be smaller than the smallest value that can be expected from
            ``rtol * abs(y)`` so that `rtol` dominates the allowable error. If `atol` is
            larger than ``rtol * abs(y)`` the number of correct digits is not guaranteed.
            Conversely, to achieve the desired `atol`, set `rtol` such that
            ``rtol * abs(y)`` is always smaller than `atol`. Default values are 1e-8 for
            `rtol` and 0 for `atol`.
        max_subdivisions : int, optional
            Upper bound on the number of subdivisions to perform. Default is 10,000.
        args : tuple, optional
            Additional positional args passed to `f`, if any.
        workers : int or map-like callable, optional
            If `workers` is an integer, part of the computation is done in parallel
            subdivided to this many tasks (using :class:`python:multiprocessing.pool.Pool`).
            Supply `-1` to use all cores available to the Process. Alternatively, supply a
            map-like callable, such as :meth:`python:multiprocessing.pool.Pool.map` for
            evaluating the population in parallel. This evaluation is carried out as
            ``workers(func, iterable)``.
        points : list of array_like, optional
            List of points to avoid evaluating `f` at, under the condition that the rule
            being used does not evaluate `f` on the boundary of a region (which is the
            case for all Genz-Malik and Gauss-Kronrod rules). This can be useful if `f` has
            a singularity at the specified point. This should be a list of array-likes where
            each element has length ``ndim``. Default is empty. See Examples.
        
        Returns
        -------
        res : object
            Object containing the results of the estimation. It has the following
            attributes:
        
            estimate : ndarray
                Estimate of the value of the integral over the overall region specified.
            error : ndarray
                Estimate of the error of the approximation over the overall region
                specified.
            status : str
                Whether the estimation was successful. Can be either: "converged",
                "not_converged".
            subdivisions : int
                Number of subdivisions performed.
            atol, rtol : float
                Requested tolerances for the approximation.
            regions: list of object
                List of objects containing the estimates of the integral over smaller
                regions of the domain.
        
            Each object in ``regions`` has the following attributes:
        
            a, b : ndarray
                Points describing the corners of the region. If the original integral
                contained infinite limits or was over a region described by `region`,
                then `a` and `b` are in the transformed coordinates.
            estimate : ndarray
                Estimate of the value of the integral over this region.
            error : ndarray
                Estimate of the error of the approximation over this region.
        
        Notes
        -----
        The algorithm uses a similar algorithm to `quad_vec`, which itself is based on the
        implementation of QUADPACK's DQAG* algorithms, implementing global error control and
        adaptive subdivision.
        
        The source of the nodes and weights used for Gauss-Kronrod quadrature can be found
        in [1]_, and the algorithm for calculating the nodes and weights in Genz-Malik
        cubature can be found in [2]_.
        
        The rules currently supported via the `rule` argument are:
        
        - ``"gauss-kronrod"``, 21-node Gauss-Kronrod
        - ``"genz-malik"``, n-node Genz-Malik
        
        If using Gauss-Kronrod for an ``n``-dim integrand where ``n > 2``, then the
        corresponding Cartesian product rule will be found by taking the Cartesian product
        of the nodes in the 1D case. This means that the number of nodes scales
        exponentially as ``21^n`` in the Gauss-Kronrod case, which may be problematic in a
        moderate number of dimensions.
        
        Genz-Malik is typically less accurate than Gauss-Kronrod but has much fewer nodes,
        so in this situation using "genz-malik" might be preferable.
        
        Infinite limits are handled with an appropriate variable transformation. Assuming
        ``a = [a_1, ..., a_n]`` and ``b = [b_1, ..., b_n]``:
        
        If :math:`a_i = -\infty` and :math:`b_i = \infty`, the i-th integration variable
        will use the transformation :math:`x = \frac{1-|t|}{t}` and :math:`t \in (-1, 1)`.
        
        If :math:`a_i \ne \pm\infty` and :math:`b_i = \infty`, the i-th integration variable
        will use the transformation :math:`x = a_i + \frac{1-t}{t}` and
        :math:`t \in (0, 1)`.
        
        If :math:`a_i = -\infty` and :math:`b_i \ne \pm\infty`, the i-th integration
        variable will use the transformation :math:`x = b_i - \frac{1-t}{t}` and
        :math:`t \in (0, 1)`.
        """
    def curve_fit(self, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=None, bounds=(-inf, inf), method=None, jac=None, full_output=False, nan_policy=None, **kwargs):
        r"""
        Use non-linear least squares to fit a function, f, to data.
        
        Assumes ``ydata = f(xdata, *params) + eps``.
        
        Parameters
        ----------
        f : callable
            The model function, f(x, ...). It must take the independent
            variable as the first argument and the parameters to fit as
            separate remaining arguments.
        xdata : array_like
            The independent variable where the data is measured.
            Should usually be an M-length sequence or an (k,M)-shaped array for
            functions with k predictors, and each element should be float
            convertible if it is an array like object.
        ydata : array_like
            The dependent data, a length M array - nominally ``f(xdata, ...)``.
        p0 : array_like, optional
            Initial guess for the parameters (length N). If None, then the
            initial values will all be 1 (if the number of parameters for the
            function can be determined using introspection, otherwise a
            ValueError is raised).
        sigma : None or scalar or M-length sequence or MxM array, optional
            Determines the uncertainty in `ydata`. If we define residuals as
            ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
            depends on its number of dimensions:
        
            - A scalar or 1-D `sigma` should contain values of standard deviations of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = sum((r / sigma) ** 2)``.
        
            - A 2-D `sigma` should contain the covariance matrix of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = r.T @ inv(sigma) @ r``.
        
              .. versionadded:: 0.19
        
            None (default) is equivalent of 1-D `sigma` filled with ones.
        absolute_sigma : bool, optional
            If True, `sigma` is used in an absolute sense and the estimated parameter
            covariance `pcov` reflects these absolute values.
        
            If False (default), only the relative magnitudes of the `sigma` values matter.
            The returned parameter covariance matrix `pcov` is based on scaling
            `sigma` by a constant factor. This constant is set by demanding that the
            reduced `chisq` for the optimal parameters `popt` when using the
            *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
            match the sample variance of the residuals after the fit. Default is False.
            Mathematically,
            ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
        check_finite : bool, optional
            If True, check that the input arrays do not contain nans of infs,
            and raise a ValueError if they do. Setting this parameter to
            False may silently produce nonsensical results if the input arrays
            do contain nans. Default is True if `nan_policy` is not specified
            explicitly and False otherwise.
        bounds : 2-tuple of array_like or `Bounds`, optional
            Lower and upper bounds on parameters. Defaults to no bounds.
            There are two ways to specify the bounds:
        
            - Instance of `Bounds` class.
        
            - 2-tuple of array_like: Each element of the tuple must be either
              an array with the length equal to the number of parameters, or a
              scalar (in which case the bound is taken to be the same for all
              parameters). Use ``np.inf`` with an appropriate sign to disable
              bounds on all or some parameters.
        
        method : {'lm', 'trf', 'dogbox'}, optional
            Method to use for optimization. See `least_squares` for more details.
            Default is 'lm' for unconstrained problems and 'trf' if `bounds` are
            provided. The method 'lm' won't work when the number of observations
            is less than the number of variables, use 'trf' or 'dogbox' in this
            case.
        
            .. versionadded:: 0.17
        jac : callable, string or None, optional
            Function with signature ``jac(x, ...)`` which computes the Jacobian
            matrix of the model function with respect to parameters as a dense
            array_like structure. It will be scaled according to provided `sigma`.
            If None (default), the Jacobian will be estimated numerically.
            String keywords for 'trf' and 'dogbox' methods can be used to select
            a finite difference scheme, see `least_squares`.
        
            .. versionadded:: 0.18
        full_output : boolean, optional
            If True, this function returns additional information: `infodict`,
            `mesg`, and `ier`.
        
            .. versionadded:: 1.9
        nan_policy : {'raise', 'omit', None}, optional
            Defines how to handle when input contains nan.
            The following options are available (default is None):
        
            * 'raise': throws an error
            * 'omit': performs the calculations ignoring nan values
            * None: no special handling of NaNs is performed
              (except what is done by check_finite); the behavior when NaNs
              are present is implementation-dependent and may change.
        
            Note that if this value is specified explicitly (not None),
            `check_finite` will be set as False.
        
            .. versionadded:: 1.11
        **kwargs
            Keyword arguments passed to `leastsq` for ``method='lm'`` or
            `least_squares` otherwise.
        
        Returns
        -------
        popt : array
            Optimal values for the parameters so that the sum of the squared
            residuals of ``f(xdata, *popt) - ydata`` is minimized.
        pcov : 2-D array
            The estimated approximate covariance of popt. The diagonals provide
            the variance of the parameter estimate. To compute one standard
            deviation errors on the parameters, use
            ``perr = np.sqrt(np.diag(pcov))``. Note that the relationship between
            `cov` and parameter error estimates is derived based on a linear
            approximation to the model function around the optimum [1]_.
            When this approximation becomes inaccurate, `cov` may not provide an
            accurate measure of uncertainty.
        
            How the `sigma` parameter affects the estimated covariance
            depends on `absolute_sigma` argument, as described above.
        
            If the Jacobian matrix at the solution doesn't have a full rank, then
            'lm' method returns a matrix filled with ``np.inf``, on the other hand
            'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
            the covariance matrix. Covariance matrices with large condition numbers
            (e.g. computed with `numpy.linalg.cond`) may indicate that results are
            unreliable.
        infodict : dict (returned only if `full_output` is True)
            a dictionary of optional outputs with the keys:
        
            ``nfev``
                The number of function calls. Methods 'trf' and 'dogbox' do not
                count function calls for numerical Jacobian approximation,
                as opposed to 'lm' method.
            ``fvec``
                The residual values evaluated at the solution, for a 1-D `sigma`
                this is ``(f(x, *popt) - ydata)/sigma``.
            ``fjac``
                A permutation of the R matrix of a QR
                factorization of the final approximate
                Jacobian matrix, stored column wise.
                Together with ipvt, the covariance of the
                estimate can be approximated.
                Method 'lm' only provides this information.
            ``ipvt``
                An integer array of length N which defines
                a permutation matrix, p, such that
                fjac*p = q*r, where r is upper triangular
                with diagonal elements of nonincreasing
                magnitude. Column j of p is column ipvt(j)
                of the identity matrix.
                Method 'lm' only provides this information.
            ``qtf``
                The vector (transpose(q) * fvec).
                Method 'lm' only provides this information.
        
            .. versionadded:: 1.9
        mesg : str (returned only if `full_output` is True)
            A string message giving information about the solution.
        
            .. versionadded:: 1.9
        ier : int (returned only if `full_output` is True)
            An integer flag. If it is equal to 1, 2, 3 or 4, the solution was
            found. Otherwise, the solution was not found. In either case, the
            optional output variable `mesg` gives more information.
        
            .. versionadded:: 1.9
        
        Raises
        ------
        ValueError
            if either `ydata` or `xdata` contain NaNs, or if incompatible options
            are used.
        
        RuntimeError
            if the least-squares minimization fails.
        
        OptimizeWarning
            if covariance of the parameters can not be estimated.
        
        See Also
        --------
        least_squares : Minimize the sum of squares of nonlinear functions.
        scipy.stats.linregress : Calculate a linear least squares regression for
                                 two sets of measurements.
        
        Notes
        -----
        Users should ensure that inputs `xdata`, `ydata`, and the output of `f`
        are ``float64``, or else the optimization may return incorrect results.
        
        With ``method='lm'``, the algorithm uses the Levenberg-Marquardt algorithm
        through `leastsq`. Note that this algorithm can only deal with
        unconstrained problems.
        
        Box constraints can be handled by methods 'trf' and 'dogbox'. Refer to
        the docstring of `least_squares` for more information.
        
        Parameters to be fitted must have similar scale. Differences of multiple
        orders of magnitude can lead to incorrect results. For the 'trf' and
        'dogbox' methods, the `x_scale` keyword argument can be used to scale
        the parameters.
        """
    def dblquad(self, a, b, gfun, hfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08):
        r"""
        Compute a double integral.
        
        Return the double (definite) integral of ``func(y, x)`` from ``x = a..b``
        and ``y = gfun(x)..hfun(x)``.
        
        Parameters
        ----------
        func : callable
            A Python function or method of at least two variables: y must be the
            first argument and x the second argument.
        a, b : float
            The limits of integration in x: `a` < `b`
        gfun : callable or float
            The lower boundary curve in y which is a function taking a single
            floating point argument (x) and returning a floating point result
            or a float indicating a constant boundary curve.
        hfun : callable or float
            The upper boundary curve in y (same requirements as `gfun`).
        args : sequence, optional
            Extra arguments to pass to `func`.
        epsabs : float, optional
            Absolute tolerance passed directly to the inner 1-D quadrature
            integration. Default is 1.49e-8. ``dblquad`` tries to obtain
            an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
            where ``i`` = inner integral of ``func(y, x)`` from ``gfun(x)``
            to ``hfun(x)``, and ``result`` is the numerical approximation.
            See `epsrel` below.
        epsrel : float, optional
            Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.
            If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
            and ``50 * (machine epsilon)``. See `epsabs` above.
        
        Returns
        -------
        y : float
            The resultant integral.
        abserr : float
            An estimate of the error.
        
        See Also
        --------
        quad : single integral
        tplquad : triple integral
        nquad : N-dimensional integrals
        fixed_quad : fixed-order Gaussian quadrature
        simpson : integrator for sampled data
        romb : integrator for sampled data
        scipy.special : for coefficients and roots of orthogonal polynomials
        
        
        Notes
        -----
        For valid results, the integral must converge; behavior for divergent
        integrals is not guaranteed.
        
        **Details of QUADPACK level routines**
        
        `quad` calls routines from the FORTRAN library QUADPACK. This section
        provides details on the conditions for each routine to be called and a
        short description of each routine. For each level of integration, ``qagse``
        is used for finite limits or ``qagie`` is used if either limit (or both!)
        are infinite. The following provides a short description from [1]_ for each
        routine.
        
        qagse
            is an integrator based on globally adaptive interval
            subdivision in connection with extrapolation, which will
            eliminate the effects of integrand singularities of
            several types.
        qagie
            handles integration over infinite intervals. The infinite range is
            mapped onto a finite interval and subsequently the same strategy as
            in ``QAGS`` is applied.
        """
    def derivative(self, x, args=(), tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0, preserve_shape=False, callback=None):
        r"""Evaluate the derivative of a elementwise, real scalar function numerically.
        
        For each element of the output of `f`, `derivative` approximates the first
        derivative of `f` at the corresponding element of `x` using finite difference
        differentiation.
        
        This function works elementwise when `x`, `step_direction`, and `args` contain
        (broadcastable) arrays.
        
        Parameters
        ----------
        f : callable
            The function whose derivative is desired. The signature must be::
        
                f(xi: ndarray, *argsi) -> ndarray
        
            where each element of ``xi`` is a finite real number and ``argsi`` is a tuple,
            which may contain an arbitrary number of arrays that are broadcastable with
            ``xi``. `f` must be an elementwise function: each scalar element ``f(xi)[j]``
            must equal ``f(xi[j])`` for valid indices ``j``. It must not mutate the array
            ``xi`` or the arrays in ``argsi``.
        x : float array_like
            Abscissae at which to evaluate the derivative. Must be broadcastable with
            `args` and `step_direction`.
        args : tuple of array_like, optional
            Additional positional array arguments to be passed to `f`. Arrays
            must be broadcastable with one another and the arrays of `init`.
            If the callable for which the root is desired requires arguments that are
            not broadcastable with `x`, wrap that callable with `f` such that `f`
            accepts only `x` and broadcastable ``*args``.
        tolerances : dictionary of floats, optional
            Absolute and relative tolerances. Valid keys of the dictionary are:
        
            - ``atol`` - absolute tolerance on the derivative
            - ``rtol`` - relative tolerance on the derivative
        
            Iteration will stop when ``res.error < atol + rtol * abs(res.df)``. The default
            `atol` is the smallest normal number of the appropriate dtype, and
            the default `rtol` is the square root of the precision of the
            appropriate dtype.
        order : int, default: 8
            The (positive integer) order of the finite difference formula to be
            used. Odd integers will be rounded up to the next even integer.
        initial_step : float array_like, default: 0.5
            The (absolute) initial step size for the finite difference derivative
            approximation.
        step_factor : float, default: 2.0
            The factor by which the step size is *reduced* in each iteration; i.e.
            the step size in iteration 1 is ``initial_step/step_factor``. If
            ``step_factor < 1``, subsequent steps will be greater than the initial
            step; this may be useful if steps smaller than some threshold are
            undesirable (e.g. due to subtractive cancellation error).
        maxiter : int, default: 10
            The maximum number of iterations of the algorithm to perform. See
            Notes.
        step_direction : integer array_like
            An array representing the direction of the finite difference steps (for
            use when `x` lies near to the boundary of the domain of the function.)
            Must be broadcastable with `x` and all `args`.
            Where 0 (default), central differences are used; where negative (e.g.
            -1), steps are non-positive; and where positive (e.g. 1), all steps are
            non-negative.
        preserve_shape : bool, default: False
            In the following, "arguments of `f`" refers to the array ``xi`` and
            any arrays within ``argsi``. Let ``shape`` be the broadcasted shape
            of `x` and all elements of `args` (which is conceptually
            distinct from ``xi` and ``argsi`` passed into `f`).
        
            - When ``preserve_shape=False`` (default), `f` must accept arguments
              of *any* broadcastable shapes.
        
            - When ``preserve_shape=True``, `f` must accept arguments of shape
              ``shape`` *or* ``shape + (n,)``, where ``(n,)`` is the number of
              abscissae at which the function is being evaluated.
        
            In either case, for each scalar element ``xi[j]`` within ``xi``, the array
            returned by `f` must include the scalar ``f(xi[j])`` at the same index.
            Consequently, the shape of the output is always the shape of the input
            ``xi``.
        
            See Examples.
        callback : callable, optional
            An optional user-supplied function to be called before the first
            iteration and after each iteration.
            Called as ``callback(res)``, where ``res`` is a ``_RichResult``
            similar to that returned by `derivative` (but containing the current
            iterate's values of all variables). If `callback` raises a
            ``StopIteration``, the algorithm will terminate immediately and
            `derivative` will return a result. `callback` must not mutate
            `res` or its attributes.
        
        Returns
        -------
        res : _RichResult
            An object similar to an instance of `scipy.optimize.OptimizeResult` with the
            following attributes. The descriptions are written as though the values will
            be scalars; however, if `f` returns an array, the outputs will be
            arrays of the same shape.
        
            success : bool array
                ``True`` where the algorithm terminated successfully (status ``0``);
                ``False`` otherwise.
            status : int array
                An integer representing the exit status of the algorithm.
        
                - ``0`` : The algorithm converged to the specified tolerances.
                - ``-1`` : The error estimate increased, so iteration was terminated.
                - ``-2`` : The maximum number of iterations was reached.
                - ``-3`` : A non-finite value was encountered.
                - ``-4`` : Iteration was terminated by `callback`.
                - ``1`` : The algorithm is proceeding normally (in `callback` only).
        
            df : float array
                The derivative of `f` at `x`, if the algorithm terminated
                successfully.
            error : float array
                An estimate of the error: the magnitude of the difference between
                the current estimate of the derivative and the estimate in the
                previous iteration.
            nit : int array
                The number of iterations of the algorithm that were performed.
            nfev : int array
                The number of points at which `f` was evaluated.
            x : float array
                The value at which the derivative of `f` was evaluated
                (after broadcasting with `args` and `step_direction`).
        
        See Also
        --------
        jacobian, hessian
        
        Notes
        -----
        The implementation was inspired by jacobi [1]_, numdifftools [2]_, and
        DERIVEST [3]_, but the implementation follows the theory of Taylor series
        more straightforwardly (and arguably naively so).
        In the first iteration, the derivative is estimated using a finite
        difference formula of order `order` with maximum step size `initial_step`.
        Each subsequent iteration, the maximum step size is reduced by
        `step_factor`, and the derivative is estimated again until a termination
        condition is reached. The error estimate is the magnitude of the difference
        between the current derivative approximation and that of the previous
        iteration.
        
        The stencils of the finite difference formulae are designed such that
        abscissae are "nested": after `f` is evaluated at ``order + 1``
        points in the first iteration, `f` is evaluated at only two new points
        in each subsequent iteration; ``order - 1`` previously evaluated function
        values required by the finite difference formula are reused, and two
        function values (evaluations at the points furthest from `x`) are unused.
        
        Step sizes are absolute. When the step size is small relative to the
        magnitude of `x`, precision is lost; for example, if `x` is ``1e20``, the
        default initial step size of ``0.5`` cannot be resolved. Accordingly,
        consider using larger initial step sizes for large magnitudes of `x`.
        
        The default tolerances are challenging to satisfy at points where the
        true derivative is exactly zero. If the derivative may be exactly zero,
        consider specifying an absolute tolerance (e.g. ``atol=1e-12``) to
        improve convergence.
        """
    def differential_evolution(self, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, rng=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1, constraints=(), x0=None, integrality=None, vectorized=False):
        r"""    
        
        
        Finds the global minimum of a multivariate function.
        
        The differential evolution method [1]_ is stochastic in nature. It does
        not use gradient methods to find the minimum, and can search large areas
        of candidate space, but often requires larger numbers of function
        evaluations than conventional gradient-based techniques.
        
        The algorithm is due to Storn and Price [2]_.
        
        Parameters
        ----------
        func : callable
            The objective function to be minimized. Must be in the form
            ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
            and ``args`` is a tuple of any additional fixed parameters needed to
            completely specify the function. The number of parameters, N, is equal
            to ``len(x)``.
        bounds : sequence or `Bounds`
            Bounds for variables. There are two ways to specify the bounds:
            
            1. Instance of `Bounds` class.
            2. ``(min, max)`` pairs for each element in ``x``, defining the
               finite lower and upper bounds for the optimizing argument of
               `func`.
            
            The total number of bounds is used to determine the number of
            parameters, N. If there are parameters whose bounds are equal the total
            number of free parameters is ``N - N_equal``.
        args : tuple, optional
            Any additional fixed parameters needed to
            completely specify the objective function.
        strategy : {str, callable}, optional
            The differential evolution strategy to use. Should be one of:
            
            - 'best1bin'
            - 'best1exp'
            - 'rand1bin'
            - 'rand1exp'
            - 'rand2bin'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'randtobest1exp'
            - 'currenttobest1bin'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'best2bin'
            
            The default is 'best1bin'. Strategies that may be implemented are
            outlined in 'Notes'.
            Alternatively the differential evolution strategy can be customized by
            providing a callable that constructs a trial vector. The callable must
            have the form ``strategy(candidate: int, population: np.ndarray, rng=None)``,
            where ``candidate`` is an integer specifying which entry of the
            population is being evolved, ``population`` is an array of shape
            ``(S, N)`` containing all the population members (where S is the
            total population size), and ``rng`` is the random number generator
            being used within the solver.
            ``candidate`` will be in the range ``[0, S)``.
            ``strategy`` must return a trial vector with shape ``(N,)``. The
            fitness of this trial vector is compared against the fitness of
            ``population[candidate]``.
            
            .. versionchanged:: 1.12.0
                Customization of evolution strategy via a callable.
        maxiter : int, optional
            The maximum number of generations over which the entire population is
            evolved. The maximum number of function evaluations (with no polishing)
            is: ``(maxiter + 1) * popsize * (N - N_equal)``
        popsize : int, optional
            A multiplier for setting the total population size. The population has
            ``popsize * (N - N_equal)`` individuals. This keyword is overridden if
            an initial population is supplied via the `init` keyword. When using
            ``init='sobol'`` the population size is calculated as the next power
            of 2 after ``popsize * (N - N_equal)``.
        tol : float, optional
            Relative tolerance for convergence, the solving stops when
            ``np.std(population_energies) <= atol + tol * np.abs(np.mean(population_energies))``,
            where and `atol` and `tol` are the absolute and relative tolerance
            respectively.
        mutation : float or tuple(float, float), optional
            The mutation constant. In the literature this is also known as
            differential weight, being denoted by :math:`F`.
            If specified as a float it should be in the range [0, 2).
            If specified as a tuple ``(min, max)`` dithering is employed. Dithering
            randomly changes the mutation constant on a generation by generation
            basis. The mutation constant for that generation is taken from
            ``U[min, max)``. Dithering can help speed convergence significantly.
            Increasing the mutation constant increases the search radius, but will
            slow down convergence.
        recombination : float, optional
            The recombination constant, should be in the range [0, 1]. In the
            literature this is also known as the crossover probability, being
            denoted by CR. Increasing this value allows a larger number of mutants
            to progress into the next generation, but at the risk of population
            stability.
        rng : {None, int, `numpy.random.Generator`}, optional
            If `rng` is passed by keyword, types other than `numpy.random.Generator` are
            passed to `numpy.random.default_rng` to instantiate a ``Generator``.
            If `rng` is already a ``Generator`` instance, then the provided instance is
            used. Specify `rng` for repeatable function behavior.
        
            If this argument is passed by position or `seed` is passed by keyword,
            legacy behavior for the argument `seed` applies:
        
            - If `seed` is None (or `numpy.random`), the `numpy.random.RandomState`
              singleton is used.
            - If `seed` is an int, a new ``RandomState`` instance is used,
              seeded with `seed`.
            - If `seed` is already a ``Generator`` or ``RandomState`` instance then
              that instance is used.
        
            .. versionchanged:: 1.15.0
                As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
                transition from use of `numpy.random.RandomState` to
                `numpy.random.Generator`, this keyword was changed from `seed` to `rng`.
                For an interim period, both keywords will continue to work, although only one
                may be specified at a time. After the interim period, function calls using the
                `seed` keyword will emit warnings. The behavior of both `seed` and
                `rng` are outlined above, but only the `rng` keyword should be used in new code.
                
        disp : bool, optional
            Prints the evaluated `func` at every iteration.
        callback : callable, optional
            A callable called after each iteration. Has the signature::
            
                callback(intermediate_result: OptimizeResult)
            
            where ``intermediate_result`` is a keyword parameter containing an
            `OptimizeResult` with attributes ``x`` and ``fun``, the best solution
            found so far and the objective function. Note that the name
            of the parameter must be ``intermediate_result`` for the callback
            to be passed an `OptimizeResult`.
            
            The callback also supports a signature like::
            
                callback(x, convergence: float=val)
            
            ``val`` represents the fractional value of the population convergence.
            When ``val`` is greater than ``1.0``, the function halts.
            
            Introspection is used to determine which of the signatures is invoked.
            
            Global minimization will halt if the callback raises ``StopIteration``
            or returns ``True``; any polishing is still carried out.
            
            .. versionchanged:: 1.12.0
                callback accepts the ``intermediate_result`` keyword.
        polish : bool, optional
            If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
            method is used to polish the best population member at the end, which
            can improve the minimization slightly. If a constrained problem is
            being studied then the `trust-constr` method is used instead. For large
            problems with many constraints, polishing can take a long time due to
            the Jacobian computations.
            
            .. versionchanged:: 1.15.0
                If `workers` is specified then the map-like callable that wraps
                `func` is supplied to `minimize` instead of it using `func`
                directly. This allows the caller to control how and where the
                invocations actually run.
        init : str or array-like, optional
            Specify which type of population initialization is performed. Should be
            one of:
            
            - 'latinhypercube'
            - 'sobol'
            - 'halton'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(S, N)``, where S is the total population size and N is
              the number of parameters.
            
            `init` is clipped to `bounds` before use.
            
            The default is 'latinhypercube'. Latin Hypercube sampling tries to
            maximize coverage of the available parameter space.
            
            'sobol' and 'halton' are superior alternatives and maximize even more
            the parameter space. 'sobol' will enforce an initial population
            size which is calculated as the next power of 2 after
            ``popsize * (N - N_equal)``. 'halton' has no requirements but is a bit
            less efficient. See `scipy.stats.qmc` for more details.
            
            'random' initializes the population randomly - this has the drawback
            that clustering can occur, preventing the whole of parameter space
            being covered. Use of an array to specify a population could be used,
            for example, to create a tight bunch of initial guesses in an location
            where the solution is known to exist, thereby reducing time for
            convergence.
        atol : float, optional
            Absolute tolerance for convergence, the solving stops when
            ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
            where and `atol` and `tol` are the absolute and relative tolerance
            respectively.
        updating : {'immediate', 'deferred'}, optional
            If ``'immediate'``, the best solution vector is continuously updated
            within a single generation [4]_. This can lead to faster convergence as
            trial vectors can take advantage of continuous improvements in the best
            solution.
            With ``'deferred'``, the best solution vector is updated once per
            generation. Only ``'deferred'`` is compatible with parallelization or
            vectorization, and the `workers` and `vectorized` keywords can
            over-ride this option.
            
            .. versionadded:: 1.2.0
        workers : int or map-like callable, optional
            If `workers` is an int the population is subdivided into `workers`
            sections and evaluated in parallel
            (uses `multiprocessing.Pool <multiprocessing>`).
            Supply -1 to use all available CPU cores.
            Alternatively supply a map-like callable, such as
            `multiprocessing.Pool.map` for evaluating the population in parallel.
            This evaluation is carried out as ``workers(func, iterable)``.
            This option will override the `updating` keyword to
            ``updating='deferred'`` if ``workers != 1``.
            This option overrides the `vectorized` keyword if ``workers != 1``.
            Requires that `func` be pickleable.
            
            .. versionadded:: 1.2.0
        constraints : {NonLinearConstraint, LinearConstraint, Bounds}
            Constraints on the solver, over and above those applied by the `bounds`
            kwd. Uses the approach by Lampinen [5]_.
            
            .. versionadded:: 1.4.0
        x0 : None or array-like, optional
            Provides an initial guess to the minimization. Once the population has
            been initialized this vector replaces the first (best) member. This
            replacement is done even if `init` is given an initial population.
            ``x0.shape == (N,)``.
            
            .. versionadded:: 1.7.0
        integrality : 1-D array, optional
            For each decision variable, a boolean value indicating whether the
            decision variable is constrained to integer values. The array is
            broadcast to ``(N,)``.
            If any decision variables are constrained to be integral, they will not
            be changed during polishing.
            Only integer values lying between the lower and upper bounds are used.
            If there are no integer values lying between the bounds then a
            `ValueError` is raised.
            
            .. versionadded:: 1.9.0
        vectorized : bool, optional
            If ``vectorized is True``, `func` is sent an `x` array with
            ``x.shape == (N, S)``, and is expected to return an array of shape
            ``(S,)``, where `S` is the number of solution vectors to be calculated.
            If constraints are applied, each of the functions used to construct
            a `Constraint` object should accept an `x` array with
            ``x.shape == (N, S)``, and return an array of shape ``(M, S)``, where
            `M` is the number of constraint components.
            This option is an alternative to the parallelization offered by
            `workers`, and may help in optimization speed by reducing interpreter
            overhead from multiple function calls. This keyword is ignored if
            ``workers != 1``.
            This option will override the `updating` keyword to
            ``updating='deferred'``.
            See the notes section for further discussion on when to use
            ``'vectorized'``, and when to use ``'workers'``.
            
            .. versionadded:: 1.9.0
        
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully,
            ``message`` which describes the cause of the termination,
            ``population`` the solution vectors present in the population, and
            ``population_energies`` the value of the objective function for each
            entry in ``population``.
            See `OptimizeResult` for a description of other attributes. If `polish`
            was employed, and a lower minimum was obtained by the polishing, then
            OptimizeResult also contains the ``jac`` attribute.
            If the eventual solution does not satisfy the applied constraints
            ``success`` will be `False`.
        
        Notes
        -----
        Differential evolution is a stochastic population based method that is
        useful for global optimization problems. At each pass through the
        population the algorithm mutates each candidate solution by mixing with
        other candidate solutions to create a trial candidate. There are several
        strategies [3]_ for creating trial candidates, which suit some problems
        more than others. The 'best1bin' strategy is a good starting point for
        many systems. In this strategy two members of the population are randomly
        chosen. Their difference is used to mutate the best member (the 'best' in
        'best1bin'), :math:`x_0`, so far:
        
        .. math::
        
            b' = x_0 + F \cdot (x_{r_0} - x_{r_1})
        
        where :math:`F` is the `mutation` parameter.
        A trial vector is then constructed. Starting with a randomly chosen ith
        parameter the trial is sequentially filled (in modulo) with parameters
        from ``b'`` or the original candidate. The choice of whether to use ``b'``
        or the original candidate is made with a binomial distribution (the 'bin'
        in 'best1bin') - a random number in [0, 1) is generated. If this number is
        less than the `recombination` constant then the parameter is loaded from
        ``b'``, otherwise it is loaded from the original candidate. The final
        parameter is always loaded from ``b'``. Once the trial candidate is built
        its fitness is assessed. If the trial is better than the original candidate
        then it takes its place. If it is also better than the best overall
        candidate it also replaces that.
        
        The other strategies available are outlined in Qiang and
        Mitchell (2014) [3]_.
        
        - ``rand1`` : :math:`b' = x_{r_0} + F \cdot (x_{r_1} - x_{r_2})`
        - ``rand2`` : :math:`b' = x_{r_0} + F \cdot (x_{r_1} + x_{r_2} - x_{r_3} - x_{r_4})`
        - ``best1`` : :math:`b' = x_0 + F \cdot (x_{r_0} - x_{r_1})`
        - ``best2`` : :math:`b' = x_0 + F \cdot (x_{r_0} + x_{r_1} - x_{r_2} - x_{r_3})`
        - ``currenttobest1`` : :math:`b' = x_i + F \cdot (x_0 - x_i + x_{r_0} - x_{r_1})`
        - ``randtobest1`` : :math:`b' = x_{r_0} + F \cdot (x_0 - x_{r_0} + x_{r_1} - x_{r_2})`
        
        where the integers :math:`r_0, r_1, r_2, r_3, r_4` are chosen randomly
        from the interval [0, NP) with `NP` being the total population size and
        the original candidate having index `i`. The user can fully customize the
        generation of the trial candidates by supplying a callable to ``strategy``.
        
        To improve your chances of finding a global minimum use higher `popsize`
        values, with higher `mutation` and (dithering), but lower `recombination`
        values. This has the effect of widening the search radius, but slowing
        convergence.
        
        By default the best solution vector is updated continuously within a single
        iteration (``updating='immediate'``). This is a modification [4]_ of the
        original differential evolution algorithm which can lead to faster
        convergence as trial vectors can immediately benefit from improved
        solutions. To use the original Storn and Price behaviour, updating the best
        solution once per iteration, set ``updating='deferred'``.
        The ``'deferred'`` approach is compatible with both parallelization and
        vectorization (``'workers'`` and ``'vectorized'`` keywords). These may
        improve minimization speed by using computer resources more efficiently.
        The ``'workers'`` distribute calculations over multiple processors. By
        default the Python `multiprocessing` module is used, but other approaches
        are also possible, such as the Message Passing Interface (MPI) used on
        clusters [6]_ [7]_. The overhead from these approaches (creating new
        Processes, etc) may be significant, meaning that computational speed
        doesn't necessarily scale with the number of processors used.
        Parallelization is best suited to computationally expensive objective
        functions. If the objective function is less expensive, then
        ``'vectorized'`` may aid by only calling the objective function once per
        iteration, rather than multiple times for all the population members; the
        interpreter overhead is reduced.
        
        .. versionadded:: 0.15.0
        """
    def direct(self, bounds: Union[Iterable, Bounds], args: tuple = (), eps: float = 0.0001, maxfun: int | None = None, maxiter: int = 1000, locally_biased: bool = True, f_min: float = -inf, f_min_rtol: float = 0.0001, vol_tol: float = 1e-16, len_tol: float = 1e-06, callback: Optional[Callable[[numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]], object]] = None) -> OptimizeResult:
        r"""
        Finds the global minimum of a function using the
        DIRECT algorithm.
        
        Parameters
        ----------
        func : callable
            The objective function to be minimized.
            ``func(x, *args) -> float``
            where ``x`` is an 1-D array with shape (n,) and ``args`` is a tuple of
            the fixed parameters needed to completely specify the function.
        bounds : sequence or `Bounds`
            Bounds for variables. There are two ways to specify the bounds:
        
            1. Instance of `Bounds` class.
            2. ``(min, max)`` pairs for each element in ``x``.
        
        args : tuple, optional
            Any additional fixed parameters needed to
            completely specify the objective function.
        eps : float, optional
            Minimal required difference of the objective function values
            between the current best hyperrectangle and the next potentially
            optimal hyperrectangle to be divided. In consequence, `eps` serves as a
            tradeoff between local and global search: the smaller, the more local
            the search becomes. Default is 1e-4.
        maxfun : int or None, optional
            Approximate upper bound on objective function evaluations.
            If `None`, will be automatically set to ``1000 * N`` where ``N``
            represents the number of dimensions. Will be capped if necessary to
            limit DIRECT's RAM usage to app. 1GiB. This will only occur for very
            high dimensional problems and excessive `max_fun`. Default is `None`.
        maxiter : int, optional
            Maximum number of iterations. Default is 1000.
        locally_biased : bool, optional
            If `True` (default), use the locally biased variant of the
            algorithm known as DIRECT_L. If `False`, use the original unbiased
            DIRECT algorithm. For hard problems with many local minima,
            `False` is recommended.
        f_min : float, optional
            Function value of the global optimum. Set this value only if the
            global optimum is known. Default is ``-np.inf``, so that this
            termination criterion is deactivated.
        f_min_rtol : float, optional
            Terminate the optimization once the relative error between the
            current best minimum `f` and the supplied global minimum `f_min`
            is smaller than `f_min_rtol`. This parameter is only used if
            `f_min` is also set. Must lie between 0 and 1. Default is 1e-4.
        vol_tol : float, optional
            Terminate the optimization once the volume of the hyperrectangle
            containing the lowest function value is smaller than `vol_tol`
            of the complete search space. Must lie between 0 and 1.
            Default is 1e-16.
        len_tol : float, optional
            If ``locally_biased=True``, terminate the optimization once half of
            the normalized maximal side length of the hyperrectangle containing
            the lowest function value is smaller than `len_tol`.
            If ``locally_biased=False``, terminate the optimization once half of
            the normalized diagonal of the hyperrectangle containing the lowest
            function value is smaller than `len_tol`. Must lie between 0 and 1.
            Default is 1e-6.
        callback : callable, optional
            A callback function with signature ``callback(xk)`` where ``xk``
            represents the best function value found so far.
        
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.
        
        Notes
        -----
        DIviding RECTangles (DIRECT) is a deterministic global
        optimization algorithm capable of minimizing a black box function with
        its variables subject to lower and upper bound constraints by sampling
        potential solutions in the search space [1]_. The algorithm starts by
        normalising the search space to an n-dimensional unit hypercube.
        It samples the function at the center of this hypercube and at 2n
        (n is the number of variables) more points, 2 in each coordinate
        direction. Using these function values, DIRECT then divides the
        domain into hyperrectangles, each having exactly one of the sampling
        points as its center. In each iteration, DIRECT chooses, using the `eps`
        parameter which defaults to 1e-4, some of the existing hyperrectangles
        to be further divided. This division process continues until either the
        maximum number of iterations or maximum function evaluations allowed
        are exceeded, or the hyperrectangle containing the minimal value found
        so far becomes small enough. If `f_min` is specified, the optimization
        will stop once this function value is reached within a relative tolerance.
        The locally biased variant of DIRECT (originally called DIRECT_L) [2]_ is
        used by default. It makes the search more locally biased and more
        efficient for cases with only a few local minima.
        
        A note about termination criteria: `vol_tol` refers to the volume of the
        hyperrectangle containing the lowest function value found so far. This
        volume decreases exponentially with increasing dimensionality of the
        problem. Therefore `vol_tol` should be decreased to avoid premature
        termination of the algorithm for higher dimensions. This does not hold
        for `len_tol`: it refers either to half of the maximal side length
        (for ``locally_biased=True``) or half of the diagonal of the
        hyperrectangle (for ``locally_biased=False``).
        
        This code is based on the DIRECT 2.0.4 Fortran code by Gablonsky et al. at
        https://ctk.math.ncsu.edu/SOFTWARE/DIRECTv204.tar.gz .
        This original version was initially converted via f2c and then cleaned up
        and reorganized by Steven G. Johnson, August 2007, for the NLopt project.
        The `direct` function wraps the C implementation.
        
        .. versionadded:: 1.9.0
        """
    def dual_annealing(self, bounds, args=(), maxiter=1000, minimizer_kwargs=None, initial_temp=5230.0, restart_temp_ratio=2e-05, visit=2.62, accept=-5.0, maxfun=10000000.0, rng=None, no_local_search=False, callback=None, x0=None):
        r"""    
        
        
        Find the global minimum of a function using Dual Annealing.
        
        Parameters
        ----------
        func : callable
            The objective function to be minimized. Must be in the form
            ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
            and ``args`` is a  tuple of any additional fixed parameters needed to
            completely specify the function.
        bounds : sequence or `Bounds`
            Bounds for variables. There are two ways to specify the bounds:
            
            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`.
        args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            objective function.
        maxiter : int, optional
            The maximum number of global search iterations. Default value is 1000.
        minimizer_kwargs : dict, optional
            Keyword arguments to be passed to the local minimizer
            (`minimize`). An important option could be ``method`` for the minimizer
            method to use.
            If no keyword arguments are provided, the local minimizer defaults to
            'L-BFGS-B' and uses the already supplied bounds. If `minimizer_kwargs`
            is specified, then the dict must contain all parameters required to
            control the local minimization. `args` is ignored in this dict, as it is
            passed automatically. `bounds` is not automatically passed on to the
            local minimizer as the method may not support them.
        initial_temp : float, optional
            The initial temperature, use higher values to facilitates a wider
            search of the energy landscape, allowing dual_annealing to escape
            local minima that it is trapped in. Default value is 5230. Range is
            (0.01, 5.e4].
        restart_temp_ratio : float, optional
            During the annealing process, temperature is decreasing, when it
            reaches ``initial_temp * restart_temp_ratio``, the reannealing process
            is triggered. Default value of the ratio is 2e-5. Range is (0, 1).
        visit : float, optional
            Parameter for visiting distribution. Default value is 2.62. Higher
            values give the visiting distribution a heavier tail, this makes
            the algorithm jump to a more distant region. The value range is (1, 3].
        accept : float, optional
            Parameter for acceptance distribution. It is used to control the
            probability of acceptance. The lower the acceptance parameter, the
            smaller the probability of acceptance. Default value is -5.0 with
            a range (-1e4, -5].
        maxfun : int, optional
            Soft limit for the number of objective function calls. If the
            algorithm is in the middle of a local search, this number will be
            exceeded, the algorithm will stop just after the local search is
            done. Default value is 1e7.
        rng : {None, int, `numpy.random.Generator`}, optional
            If `rng` is passed by keyword, types other than `numpy.random.Generator` are
            passed to `numpy.random.default_rng` to instantiate a ``Generator``.
            If `rng` is already a ``Generator`` instance, then the provided instance is
            used. Specify `rng` for repeatable function behavior.
        
            If this argument is passed by position or `seed` is passed by keyword,
            legacy behavior for the argument `seed` applies:
        
            - If `seed` is None (or `numpy.random`), the `numpy.random.RandomState`
              singleton is used.
            - If `seed` is an int, a new ``RandomState`` instance is used,
              seeded with `seed`.
            - If `seed` is already a ``Generator`` or ``RandomState`` instance then
              that instance is used.
        
            .. versionchanged:: 1.15.0
                As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
                transition from use of `numpy.random.RandomState` to
                `numpy.random.Generator`, this keyword was changed from `seed` to `rng`.
                For an interim period, both keywords will continue to work, although only one
                may be specified at a time. After the interim period, function calls using the
                `seed` keyword will emit warnings. The behavior of both `seed` and
                `rng` are outlined above, but only the `rng` keyword should be used in new code.
                
            Specify `rng` for repeatable minimizations. The random numbers
            generated only affect the visiting distribution function
            and new coordinates generation.
        no_local_search : bool, optional
            If `no_local_search` is set to True, a traditional Generalized
            Simulated Annealing will be performed with no local search
            strategy applied.
        callback : callable, optional
            A callback function with signature ``callback(x, f, context)``,
            which will be called for all minima found.
            ``x`` and ``f`` are the coordinates and function value of the
            latest minimum found, and ``context`` has one of the following
            values:
            
            - ``0``: minimum detected in the annealing process.
            - ``1``: detection occurred in the local search process.
            - ``2``: detection done in the dual annealing process.
            
            If the callback implementation returns True, the algorithm will stop.
        x0 : ndarray, shape(n,), optional
            Coordinates of a single N-D starting point.
        
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            Important attributes are: ``x`` the solution array, ``fun`` the value
            of the function at the solution, and ``message`` which describes the
            cause of the termination.
            See `OptimizeResult` for a description of other attributes.
        
        Notes
        -----
        This function implements the Dual Annealing optimization. This stochastic
        approach derived from [3]_ combines the generalization of CSA (Classical
        Simulated Annealing) and FSA (Fast Simulated Annealing) [1]_ [2]_ coupled
        to a strategy for applying a local search on accepted locations [4]_.
        An alternative implementation of this same algorithm is described in [5]_
        and benchmarks are presented in [6]_. This approach introduces an advanced
        method to refine the solution found by the generalized annealing
        process. This algorithm uses a distorted Cauchy-Lorentz visiting
        distribution, with its shape controlled by the parameter :math:`q_{v}`
        
        .. math::
        
            g_{q_{v}}(\Delta x(t)) \propto \frac{ \
            \left[T_{q_{v}}(t) \right]^{-\frac{D}{3-q_{v}}}}{ \
            \left[{1+(q_{v}-1)\frac{(\Delta x(t))^{2}} { \
            \left[T_{q_{v}}(t)\right]^{\frac{2}{3-q_{v}}}}}\right]^{ \
            \frac{1}{q_{v}-1}+\frac{D-1}{2}}}
        
        Where :math:`t` is the artificial time. This visiting distribution is used
        to generate a trial jump distance :math:`\Delta x(t)` of variable
        :math:`x(t)` under artificial temperature :math:`T_{q_{v}}(t)`.
        
        From the starting point, after calling the visiting distribution
        function, the acceptance probability is computed as follows:
        
        .. math::
        
            p_{q_{a}} = \min{\{1,\left[1-(1-q_{a}) \beta \Delta E \right]^{ \
            \frac{1}{1-q_{a}}}\}}
        
        Where :math:`q_{a}` is a acceptance parameter. For :math:`q_{a}<1`, zero
        acceptance probability is assigned to the cases where
        
        .. math::
        
            [1-(1-q_{a}) \beta \Delta E] < 0
        
        The artificial temperature :math:`T_{q_{v}}(t)` is decreased according to
        
        .. math::
        
            T_{q_{v}}(t) = T_{q_{v}}(1) \frac{2^{q_{v}-1}-1}{\left( \
            1 + t\right)^{q_{v}-1}-1}
        
        Where :math:`q_{v}` is the visiting parameter.
        
        .. versionadded:: 1.2.0
        """
    def find_minimum(self, init, args=(), tolerances=None, maxiter=100, callback=None):
        r"""Find the minimum of an unimodal, real-valued function of a real variable.
        
        For each element of the output of `f`, `find_minimum` seeks the scalar minimizer
        that minimizes the element. This function currently uses Chandrupatla's
        bracketing minimization algorithm [1]_ and therefore requires argument `init`
        to provide a three-point minimization bracket: ``x1 < x2 < x3`` such that
        ``func(x1) >= func(x2) <= func(x3)``, where one of the inequalities is strict.
        
        Provided a valid bracket, `find_minimum` is guaranteed to converge to a local
        minimum that satisfies the provided `tolerances` if the function is continuous
        within the bracket.
        
        This function works elementwise when `init` and `args` contain (broadcastable)
        arrays.
        
        Parameters
        ----------
        f : callable
            The function whose minimizer is desired. The signature must be::
        
                f(x: array, *args) -> array
        
            where each element of ``x`` is a finite real and ``args`` is a tuple,
            which may contain an arbitrary number of arrays that are broadcastable
            with ``x``.
        
            `f` must be an elementwise function: each element ``f(x)[i]``
            must equal ``f(x[i])`` for all indices ``i``. It must not mutate the
            array ``x`` or the arrays in ``args``.
        
            `find_minimum` seeks an array ``x`` such that ``f(x)`` is an array of
            local minima.
        init : 3-tuple of float array_like
            The abscissae of a standard scalar minimization bracket. A bracket is
            valid if arrays ``x1, x2, x3 = init`` satisfy ``x1 < x2 < x3`` and
            ``func(x1) >= func(x2) <= func(x3)``, where one of the inequalities
            is strict. Arrays must be broadcastable with one another and the arrays
            of `args`.
        args : tuple of array_like, optional
            Additional positional array arguments to be passed to `f`. Arrays
            must be broadcastable with one another and the arrays of `init`.
            If the callable for which the root is desired requires arguments that are
            not broadcastable with `x`, wrap that callable with `f` such that `f`
            accepts only `x` and broadcastable ``*args``.
        tolerances : dictionary of floats, optional
            Absolute and relative tolerances on the root and function value.
            Valid keys of the dictionary are:
        
            - ``xatol`` - absolute tolerance on the root
            - ``xrtol`` - relative tolerance on the root
            - ``fatol`` - absolute tolerance on the function value
            - ``frtol`` - relative tolerance on the function value
        
            See Notes for default values and explicit termination conditions.
        maxiter : int, default: 100
            The maximum number of iterations of the algorithm to perform.
        callback : callable, optional
            An optional user-supplied function to be called before the first
            iteration and after each iteration.
            Called as ``callback(res)``, where ``res`` is a ``_RichResult``
            similar to that returned by `find_minimum` (but containing the current
            iterate's values of all variables). If `callback` raises a
            ``StopIteration``, the algorithm will terminate immediately and
            `find_root` will return a result. `callback` must not mutate
            `res` or its attributes.
        
        Returns
        -------
        res : _RichResult
            An object similar to an instance of `scipy.optimize.OptimizeResult` with the
            following attributes. The descriptions are written as though the values will
            be scalars; however, if `f` returns an array, the outputs will be
            arrays of the same shape.
        
            success : bool array
                ``True`` where the algorithm terminated successfully (status ``0``);
                ``False`` otherwise.
            status : int array
                An integer representing the exit status of the algorithm.
        
                - ``0`` : The algorithm converged to the specified tolerances.
                - ``-1`` : The algorithm encountered an invalid bracket.
                - ``-2`` : The maximum number of iterations was reached.
                - ``-3`` : A non-finite value was encountered.
                - ``-4`` : Iteration was terminated by `callback`.
                - ``1`` : The algorithm is proceeding normally (in `callback` only).
        
            x : float array
                The minimizer of the function, if the algorithm terminated successfully.
            f_x : float array
                The value of `f` evaluated at `x`.
            nfev : int array
                The number of abscissae at which `f` was evaluated to find the root.
                This is distinct from the number of times `f` is *called* because the
                the function may evaluated at multiple points in a single call.
            nit : int array
                The number of iterations of the algorithm that were performed.
            bracket : tuple of float arrays
                The final three-point bracket.
            f_bracket : tuple of float arrays
                The value of `f` evaluated at the bracket points.
        
        Notes
        -----
        Implemented based on Chandrupatla's original paper [1]_.
        
        If ``xl < xm < xr`` are the points of the bracket and ``fl >= fm <= fr``
        (where one of the inequalities is strict) are the values of `f` evaluated
        at those points, then the algorithm is considered to have converged when:
        
        - ``xr - xl <= abs(xm)*xrtol + xatol`` or
        - ``(fl - 2*fm + fr)/2 <= abs(fm)*frtol + fatol``.
        
        Note that first of these differs from the termination conditions described
        in [1]_.
        
        The default value of `xrtol` is the square root of the precision of the
        appropriate dtype, and ``xatol = fatol = frtol`` is the smallest normal
        number of the appropriate dtype.
        """
    def find_root(self, init, args=(), tolerances=None, maxiter=None, callback=None):
        r"""Find the root of a monotonic, real-valued function of a real variable.
        
        For each element of the output of `f`, `find_root` seeks the scalar
        root that makes the element 0. This function currently uses Chandrupatla's
        bracketing algorithm [1]_ and therefore requires argument `init` to
        provide a bracket around the root: the function values at the two endpoints
        must have opposite signs.
        
        Provided a valid bracket, `find_root` is guaranteed to converge to a solution
        that satisfies the provided `tolerances` if the function is continuous within
        the bracket.
        
        This function works elementwise when `init` and `args` contain (broadcastable)
        arrays.
        
        Parameters
        ----------
        f : callable
            The function whose root is desired. The signature must be::
        
                f(x: array, *args) -> array
        
            where each element of ``x`` is a finite real and ``args`` is a tuple,
            which may contain an arbitrary number of arrays that are broadcastable
            with ``x``.
        
            `f` must be an elementwise function: each element ``f(x)[i]``
            must equal ``f(x[i])`` for all indices ``i``. It must not mutate the
            array ``x`` or the arrays in ``args``.
        
            `find_root` seeks an array ``x`` such that ``f(x)`` is an array of zeros.
        init : 2-tuple of float array_like
            The lower and upper endpoints of a bracket surrounding the desired root.
            A bracket is valid if arrays ``xl, xr = init`` satisfy ``xl < xr`` and
            ``sign(f(xl)) == -sign(f(xr))`` elementwise. Arrays be broadcastable with
            one another and `args`.
        args : tuple of array_like, optional
            Additional positional array arguments to be passed to `f`. Arrays
            must be broadcastable with one another and the arrays of `init`.
            If the callable for which the root is desired requires arguments that are
            not broadcastable with `x`, wrap that callable with `f` such that `f`
            accepts only `x` and broadcastable ``*args``.
        tolerances : dictionary of floats, optional
            Absolute and relative tolerances on the root and function value.
            Valid keys of the dictionary are:
        
            - ``xatol`` - absolute tolerance on the root
            - ``xrtol`` - relative tolerance on the root
            - ``fatol`` - absolute tolerance on the function value
            - ``frtol`` - relative tolerance on the function value
        
            See Notes for default values and explicit termination conditions.
        maxiter : int, optional
            The maximum number of iterations of the algorithm to perform.
            The default is the maximum possible number of bisections within
            the (normal) floating point numbers of the relevant dtype.
        callback : callable, optional
            An optional user-supplied function to be called before the first
            iteration and after each iteration.
            Called as ``callback(res)``, where ``res`` is a ``_RichResult``
            similar to that returned by `find_root` (but containing the current
            iterate's values of all variables). If `callback` raises a
            ``StopIteration``, the algorithm will terminate immediately and
            `find_root` will return a result. `callback` must not mutate
            `res` or its attributes.
        
        Returns
        -------
        res : _RichResult
            An object similar to an instance of `scipy.optimize.OptimizeResult` with the
            following attributes. The descriptions are written as though the values will
            be scalars; however, if `f` returns an array, the outputs will be
            arrays of the same shape.
        
            success : bool array
                ``True`` where the algorithm terminated successfully (status ``0``);
                ``False`` otherwise.
            status : int array
                An integer representing the exit status of the algorithm.
        
                - ``0`` : The algorithm converged to the specified tolerances.
                - ``-1`` : The initial bracket was invalid.
                - ``-2`` : The maximum number of iterations was reached.
                - ``-3`` : A non-finite value was encountered.
                - ``-4`` : Iteration was terminated by `callback`.
                - ``1`` : The algorithm is proceeding normally (in `callback` only).
        
            x : float array
                The root of the function, if the algorithm terminated successfully.
            f_x : float array
                The value of `f` evaluated at `x`.
            nfev : int array
                The number of abscissae at which `f` was evaluated to find the root.
                This is distinct from the number of times `f` is *called* because the
                the function may evaluated at multiple points in a single call.
            nit : int array
                The number of iterations of the algorithm that were performed.
            bracket : tuple of float arrays
                The lower and upper endpoints of the final bracket.
            f_bracket : tuple of float arrays
                The value of `f` evaluated at the lower and upper endpoints of the
                bracket.
        
        Notes
        -----
        Implemented based on Chandrupatla's original paper [1]_.
        
        Let:
        
        -  ``a, b = init`` be the left and right endpoints of the initial bracket,
        - ``xl`` and ``xr`` be the left and right endpoints of the final bracket,
        - ``xmin = xl if abs(f(xl)) <= abs(f(xr)) else xr`` be the final bracket
          endpoint with the smaller function value, and
        - ``fmin0 = min(f(a), f(b))`` be the minimum of the two values of the
          function evaluated at the initial bracket endpoints.
        
        Then the algorithm is considered to have converged when
        
        - ``abs(xr - xl) < xatol + abs(xmin) * xrtol`` or
        - ``fun(xmin) <= fatol + abs(fmin0) * frtol``.
        
        This is equivalent to the termination condition described in [1]_ with
        ``xrtol = 4e-10``, ``xatol = 1e-5``, and ``fatol = frtol = 0``.
        However, the default values of the `tolerances` dictionary are
        ``xatol = 4*tiny``, ``xrtol = 4*eps``, ``frtol = 0``, and ``fatol = tiny``,
        where ``eps`` and ``tiny`` are the precision and smallest normal number
        of the result ``dtype`` of function inputs and outputs.
        """
    def fixed_point(self, x0, args=(), xtol=1e-08, maxiter=500, method='del2'):
        r"""
        Find a fixed point of the function.
        
        Given a function of one or more variables and a starting point, find a
        fixed point of the function: i.e., where ``func(x0) == x0``.
        
        Parameters
        ----------
        func : function
            Function to evaluate.
        x0 : array_like
            Fixed point of function.
        args : tuple, optional
            Extra arguments to `func`.
        xtol : float, optional
            Convergence tolerance, defaults to 1e-08.
        maxiter : int, optional
            Maximum number of iterations, defaults to 500.
        method : {"del2", "iteration"}, optional
            Method of finding the fixed-point, defaults to "del2",
            which uses Steffensen's Method with Aitken's ``Del^2``
            convergence acceleration [1]_. The "iteration" method simply iterates
            the function until convergence is detected, without attempting to
            accelerate the convergence.
        """
    def fixed_quad(self, a, b, args=(), n=5):
        r"""
        Compute a definite integral using fixed-order Gaussian quadrature.
        
        Integrate `func` from `a` to `b` using Gaussian quadrature of
        order `n`.
        
        Parameters
        ----------
        func : callable
            A Python function or method to integrate (must accept vector inputs).
            If integrating a vector-valued function, the returned array must have
            shape ``(..., len(x))``.
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.
        args : tuple, optional
            Extra arguments to pass to function, if any.
        n : int, optional
            Order of quadrature integration. Default is 5.
        
        Returns
        -------
        val : float
            Gaussian quadrature approximation to the integral
        none : None
            Statically returned value of None
        
        See Also
        --------
        quad : adaptive quadrature using QUADPACK
        dblquad : double integrals
        tplquad : triple integrals
        romb : integrators for sampled data
        simpson : integrators for sampled data
        cumulative_trapezoid : cumulative integration for sampled data
        """
    def hessian(self, x, tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0):
        r"""Evaluate the Hessian of a function numerically.
        
        Parameters
        ----------
        f : callable
            The function whose Hessian is desired. The signature must be::
        
                f(xi: ndarray) -> ndarray
        
            where each element of ``xi`` is a finite real. If the function to be
            differentiated accepts additional arguments, wrap it (e.g. using
            `functools.partial` or ``lambda``) and pass the wrapped callable
            into `hessian`. `f` must not mutate the array ``xi``. See Notes
            regarding vectorization and the dimensionality of the input and output.
        x : float array_like
            Points at which to evaluate the Hessian. Must have at least one dimension.
            See Notes regarding the dimensionality and vectorization.
        tolerances : dictionary of floats, optional
            Absolute and relative tolerances. Valid keys of the dictionary are:
        
            - ``atol`` - absolute tolerance on the derivative
            - ``rtol`` - relative tolerance on the derivative
        
            Iteration will stop when ``res.error < atol + rtol * abs(res.df)``. The default
            `atol` is the smallest normal number of the appropriate dtype, and
            the default `rtol` is the square root of the precision of the
            appropriate dtype.
        order : int, default: 8
            The (positive integer) order of the finite difference formula to be
            used. Odd integers will be rounded up to the next even integer.
        initial_step : float, default: 0.5
            The (absolute) initial step size for the finite difference derivative
            approximation.
        step_factor : float, default: 2.0
            The factor by which the step size is *reduced* in each iteration; i.e.
            the step size in iteration 1 is ``initial_step/step_factor``. If
            ``step_factor < 1``, subsequent steps will be greater than the initial
            step; this may be useful if steps smaller than some threshold are
            undesirable (e.g. due to subtractive cancellation error).
        maxiter : int, default: 10
            The maximum number of iterations of the algorithm to perform. See
            Notes.
        
        Returns
        -------
        res : _RichResult
            An object similar to an instance of `scipy.optimize.OptimizeResult` with the
            following attributes. The descriptions are written as though the values will
            be scalars; however, if `f` returns an array, the outputs will be
            arrays of the same shape.
        
            success : bool array
                ``True`` where the algorithm terminated successfully (status ``0``);
                ``False`` otherwise.
            status : int array
                An integer representing the exit status of the algorithm.
        
                - ``0`` : The algorithm converged to the specified tolerances.
                - ``-1`` : The error estimate increased, so iteration was terminated.
                - ``-2`` : The maximum number of iterations was reached.
                - ``-3`` : A non-finite value was encountered.
        
            ddf : float array
                The Hessian of `f` at `x`, if the algorithm terminated
                successfully.
            error : float array
                An estimate of the error: the magnitude of the difference between
                the current estimate of the Hessian and the estimate in the
                previous iteration.
            nfev : int array
                The number of points at which `f` was evaluated.
        
            Each element of an attribute is associated with the corresponding
            element of `ddf`. For instance, element ``[i, j]`` of `nfev` is the
            number of points at which `f` was evaluated for the sake of
            computing element ``[i, j]`` of `ddf`.
        
        See Also
        --------
        derivative, jacobian
        
        Notes
        -----
        Suppose we wish to evaluate the Hessian of a function
        :math:`f: \mathbf{R}^m \rightarrow \mathbf{R}`, and we assign to variable
        ``m`` the positive integer value of :math:`m`. If we wish to evaluate
        the Hessian at a single point, then:
        
        - argument `x` must be an array of shape ``(m,)``
        - argument `f` must be vectorized to accept an array of shape
          ``(m, ...)``. The first axis represents the :math:`m` inputs of
          :math:`f`; the remaining axes indicated by ellipses are for evaluating
          the function at several abscissae in a single call.
        - argument `f` must return an array of shape ``(...)``.
        - attribute ``dff`` of the result object will be an array of shape ``(m, m)``,
          the Hessian.
        
        This function is also vectorized in the sense that the Hessian can be
        evaluated at ``k`` points in a single call. In this case, `x` would be an
        array of shape ``(m, k)``, `f` would accept an array of shape
        ``(m, ...)`` and return an array of shape ``(...)``, and the ``ddf``
        attribute of the result would have shape ``(m, m, k)``. Note that the
        axis associated with the ``k`` points is included within the axes
        denoted by ``(...)``.
        
        Currently, `hessian` is implemented by nesting calls to `jacobian`.
        All options passed to `hessian` are used for both the inner and outer
        calls with one exception: the `rtol` used in the inner `jacobian` call
        is tightened by a factor of 100 with the expectation that the inner
        error can be ignored. A consequence is that `rtol` should not be set
        less than 100 times the precision of the dtype of `x`; a warning is
        emitted otherwise.
        """
    def jacobian(self, x, tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0):
        r"""Evaluate the Jacobian of a function numerically.
        
        Parameters
        ----------
        f : callable
            The function whose Jacobian is desired. The signature must be::
        
                f(xi: ndarray) -> ndarray
        
            where each element of ``xi`` is a finite real. If the function to be
            differentiated accepts additional arguments, wrap it (e.g. using
            `functools.partial` or ``lambda``) and pass the wrapped callable
            into `jacobian`. `f` must not mutate the array ``xi``. See Notes
            regarding vectorization and the dimensionality of the input and output.
        x : float array_like
            Points at which to evaluate the Jacobian. Must have at least one dimension.
            See Notes regarding the dimensionality and vectorization.
        tolerances : dictionary of floats, optional
            Absolute and relative tolerances. Valid keys of the dictionary are:
        
            - ``atol`` - absolute tolerance on the derivative
            - ``rtol`` - relative tolerance on the derivative
        
            Iteration will stop when ``res.error < atol + rtol * abs(res.df)``. The default
            `atol` is the smallest normal number of the appropriate dtype, and
            the default `rtol` is the square root of the precision of the
            appropriate dtype.
        maxiter : int, default: 10
            The maximum number of iterations of the algorithm to perform. See
            Notes.
        order : int, default: 8
            The (positive integer) order of the finite difference formula to be
            used. Odd integers will be rounded up to the next even integer.
        initial_step : float array_like, default: 0.5
            The (absolute) initial step size for the finite difference derivative
            approximation. Must be broadcastable with `x` and `step_direction`.
        step_factor : float, default: 2.0
            The factor by which the step size is *reduced* in each iteration; i.e.
            the step size in iteration 1 is ``initial_step/step_factor``. If
            ``step_factor < 1``, subsequent steps will be greater than the initial
            step; this may be useful if steps smaller than some threshold are
            undesirable (e.g. due to subtractive cancellation error).
        step_direction : integer array_like
            An array representing the direction of the finite difference steps (e.g.
            for use when `x` lies near to the boundary of the domain of the function.)
            Must be broadcastable with `x` and `initial_step`.
            Where 0 (default), central differences are used; where negative (e.g.
            -1), steps are non-positive; and where positive (e.g. 1), all steps are
            non-negative.
        
        Returns
        -------
        res : _RichResult
            An object similar to an instance of `scipy.optimize.OptimizeResult` with the
            following attributes. The descriptions are written as though the values will
            be scalars; however, if `f` returns an array, the outputs will be
            arrays of the same shape.
        
            success : bool array
                ``True`` where the algorithm terminated successfully (status ``0``);
                ``False`` otherwise.
            status : int array
                An integer representing the exit status of the algorithm.
        
                - ``0`` : The algorithm converged to the specified tolerances.
                - ``-1`` : The error estimate increased, so iteration was terminated.
                - ``-2`` : The maximum number of iterations was reached.
                - ``-3`` : A non-finite value was encountered.
        
            df : float array
                The Jacobian of `f` at `x`, if the algorithm terminated
                successfully.
            error : float array
                An estimate of the error: the magnitude of the difference between
                the current estimate of the Jacobian and the estimate in the
                previous iteration.
            nit : int array
                The number of iterations of the algorithm that were performed.
            nfev : int array
                The number of points at which `f` was evaluated.
        
            Each element of an attribute is associated with the corresponding
            element of `df`. For instance, element ``i`` of `nfev` is the
            number of points at which `f` was evaluated for the sake of
            computing element ``i`` of `df`.
        
        See Also
        --------
        derivative, hessian
        
        Notes
        -----
        Suppose we wish to evaluate the Jacobian of a function
        :math:`f: \mathbf{R}^m \rightarrow \mathbf{R}^n`. Assign to variables
        ``m`` and ``n`` the positive integer values of :math:`m` and :math:`n`,
        respectively, and let ``...`` represent an arbitrary tuple of integers.
        If we wish to evaluate the Jacobian at a single point, then:
        
        - argument `x` must be an array of shape ``(m,)``
        - argument `f` must be vectorized to accept an array of shape ``(m, ...)``.
          The first axis represents the :math:`m` inputs of :math:`f`; the remainder
          are for evaluating the function at multiple points in a single call.
        - argument `f` must return an array of shape ``(n, ...)``. The first
          axis represents the :math:`n` outputs of :math:`f`; the remainder
          are for the result of evaluating the function at multiple points.
        - attribute ``df`` of the result object will be an array of shape ``(n, m)``,
          the Jacobian.
        
        This function is also vectorized in the sense that the Jacobian can be
        evaluated at ``k`` points in a single call. In this case, `x` would be an
        array of shape ``(m, k)``, `f` would accept an array of shape
        ``(m, k, ...)`` and return an array of shape ``(n, k, ...)``, and the ``df``
        attribute of the result would have shape ``(n, m, k)``.
        
        Suppose the desired callable ``f_not_vectorized`` is not vectorized; it can
        only accept an array of shape ``(m,)``. A simple solution to satisfy the required
        interface is to wrap ``f_not_vectorized`` as follows::
        
            def f(x):
                return np.apply_along_axis(f_not_vectorized, axis=0, arr=x)
        
        Alternatively, suppose the desired callable ``f_vec_q`` is vectorized, but
        only for 2-D arrays of shape ``(m, q)``. To satisfy the required interface,
        consider::
        
            def f(x):
                m, batch = x.shape[0], x.shape[1:]  # x.shape is (m, ...)
                x = np.reshape(x, (m, -1))  # `-1` is short for q = prod(batch)
                res = f_vec_q(x)  # pass shape (m, q) to function
                n = res.shape[0]
                return np.reshape(res, (n,) + batch)  # return shape (n, ...)
        
        Then pass the wrapped callable ``f`` as the first argument of `jacobian`.
        """
    def least_squares(self, x0, jac='2-point', bounds=(-inf, inf), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options=None, jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs=None):
        r"""Solve a nonlinear least-squares problem with bounds on the variables.
        
        Given the residuals f(x) (an m-D real function of n real
        variables) and the loss function rho(s) (a scalar function), `least_squares`
        finds a local minimum of the cost function F(x)::
        
            minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)
            subject to lb <= x <= ub
        
        The purpose of the loss function rho(s) is to reduce the influence of
        outliers on the solution.
        
        Parameters
        ----------
        fun : callable
            Function which computes the vector of residuals, with the signature
            ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with
            respect to its first argument. The argument ``x`` passed to this
            function is an ndarray of shape (n,) (never a scalar, even for n=1).
            It must allocate and return a 1-D array_like of shape (m,) or a scalar.
            If the argument ``x`` is complex or the function ``fun`` returns
            complex residuals, it must be wrapped in a real function of real
            arguments, as shown at the end of the Examples section.
        x0 : array_like with shape (n,) or float
            Initial guess on independent variables. If float, it will be treated
            as a 1-D array with one element. When `method` is 'trf', the initial
            guess might be slightly adjusted to lie sufficiently within the given
            `bounds`.
        jac : {'2-point', '3-point', 'cs', callable}, optional
            Method of computing the Jacobian matrix (an m-by-n matrix, where
            element (i, j) is the partial derivative of f[i] with respect to
            x[j]). The keywords select a finite difference scheme for numerical
            estimation. The scheme '3-point' is more accurate, but requires
            twice as many operations as '2-point' (default). The scheme 'cs'
            uses complex steps, and while potentially the most accurate, it is
            applicable only when `fun` correctly handles complex inputs and
            can be analytically continued to the complex plane. Method 'lm'
            always uses the '2-point' scheme. If callable, it is used as
            ``jac(x, *args, **kwargs)`` and should return a good approximation
            (or the exact value) for the Jacobian as an array_like (np.atleast_2d
            is applied), a sparse matrix (csr_matrix preferred for performance) or
            a `scipy.sparse.linalg.LinearOperator`.
        bounds : 2-tuple of array_like or `Bounds`, optional
            There are two ways to specify bounds:
        
            1. Instance of `Bounds` class
            2. Lower and upper bounds on independent variables. Defaults to no
               bounds. Each array must match the size of `x0` or be a scalar,
               in the latter case a bound will be the same for all variables.
               Use ``np.inf`` with an appropriate sign to disable bounds on all
               or some variables.
        
        method : {'trf', 'dogbox', 'lm'}, optional
            Algorithm to perform minimization.
        
            * 'trf' : Trust Region Reflective algorithm, particularly suitable
              for large sparse problems with bounds. Generally robust method.
            * 'dogbox' : dogleg algorithm with rectangular trust regions,
              typical use case is small problems with bounds. Not recommended
              for problems with rank-deficient Jacobian.
            * 'lm' : Levenberg-Marquardt algorithm as implemented in MINPACK.
              Doesn't handle bounds and sparse Jacobians. Usually the most
              efficient method for small unconstrained problems.
        
            Default is 'trf'. See Notes for more information.
        ftol : float or None, optional
            Tolerance for termination by the change of the cost function. Default
            is 1e-8. The optimization process is stopped when ``dF < ftol * F``,
            and there was an adequate agreement between a local quadratic model and
            the true model in the last step.
        
            If None and 'method' is not 'lm', the termination by this condition is
            disabled. If 'method' is 'lm', this tolerance must be higher than
            machine epsilon.
        xtol : float or None, optional
            Tolerance for termination by the change of the independent variables.
            Default is 1e-8. The exact condition depends on the `method` used:
        
            * For 'trf' and 'dogbox' : ``norm(dx) < xtol * (xtol + norm(x))``.
            * For 'lm' : ``Delta < xtol * norm(xs)``, where ``Delta`` is
              a trust-region radius and ``xs`` is the value of ``x``
              scaled according to `x_scale` parameter (see below).
        
            If None and 'method' is not 'lm', the termination by this condition is
            disabled. If 'method' is 'lm', this tolerance must be higher than
            machine epsilon.
        gtol : float or None, optional
            Tolerance for termination by the norm of the gradient. Default is 1e-8.
            The exact condition depends on a `method` used:
        
            * For 'trf' : ``norm(g_scaled, ord=np.inf) < gtol``, where
              ``g_scaled`` is the value of the gradient scaled to account for
              the presence of the bounds [STIR]_.
            * For 'dogbox' : ``norm(g_free, ord=np.inf) < gtol``, where
              ``g_free`` is the gradient with respect to the variables which
              are not in the optimal state on the boundary.
            * For 'lm' : the maximum absolute value of the cosine of angles
              between columns of the Jacobian and the residual vector is less
              than `gtol`, or the residual vector is zero.
        
            If None and 'method' is not 'lm', the termination by this condition is
            disabled. If 'method' is 'lm', this tolerance must be higher than
            machine epsilon.
        x_scale : array_like or 'jac', optional
            Characteristic scale of each variable. Setting `x_scale` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along jth
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting `x_scale` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to 'jac', the scale is iteratively updated using the
            inverse norms of the columns of the Jacobian matrix (as described in
            [JJMore]_).
        loss : str or callable, optional
            Determines the loss function. The following keyword values are allowed:
        
            * 'linear' (default) : ``rho(z) = z``. Gives a standard
              least-squares problem.
            * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
              approximation of l1 (absolute value) loss. Usually a good
              choice for robust least squares.
            * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
              similarly to 'soft_l1'.
            * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
              influence, but may cause difficulties in optimization process.
            * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
              a single residual, has properties similar to 'cauchy'.
        
            If callable, it must take a 1-D ndarray ``z=f**2`` and return an
            array_like with shape (3, m) where row 0 contains function values,
            row 1 contains first derivatives and row 2 contains second
            derivatives. Method 'lm' supports only 'linear' loss.
        f_scale : float, optional
            Value of soft margin between inlier and outlier residuals, default
            is 1.0. The loss function is evaluated as follows
            ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,
            and ``rho`` is determined by `loss` parameter. This parameter has
            no effect with ``loss='linear'``, but for other `loss` values it is
            of crucial importance.
        max_nfev : None or int, optional
            Maximum number of function evaluations before the termination.
            If None (default), the value is chosen automatically:
        
            * For 'trf' and 'dogbox' : 100 * n.
            * For 'lm' :  100 * n if `jac` is callable and 100 * n * (n + 1)
              otherwise (because 'lm' counts function calls in Jacobian
              estimation).
        
        diff_step : None or array_like, optional
            Determines the relative step size for the finite difference
            approximation of the Jacobian. The actual step is computed as
            ``x * diff_step``. If None (default), then `diff_step` is taken to be
            a conventional "optimal" power of machine epsilon for the finite
            difference scheme used [NR]_.
        tr_solver : {None, 'exact', 'lsmr'}, optional
            Method for solving trust-region subproblems, relevant only for 'trf'
            and 'dogbox' methods.
        
            * 'exact' is suitable for not very large problems with dense
              Jacobian matrices. The computational complexity per iteration is
              comparable to a singular value decomposition of the Jacobian
              matrix.
            * 'lsmr' is suitable for problems with sparse and large Jacobian
              matrices. It uses the iterative procedure
              `scipy.sparse.linalg.lsmr` for finding a solution of a linear
              least-squares problem and only requires matrix-vector product
              evaluations.
        
            If None (default), the solver is chosen based on the type of Jacobian
            returned on the first iteration.
        tr_options : dict, optional
            Keyword options passed to trust-region solver.
        
            * ``tr_solver='exact'``: `tr_options` are ignored.
            * ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.
              Additionally,  ``method='trf'`` supports  'regularize' option
              (bool, default is True), which adds a regularization term to the
              normal equation, which improves convergence if the Jacobian is
              rank-deficient [Byrd]_ (eq. 3.4).
        
        jac_sparsity : {None, array_like, sparse matrix}, optional
            Defines the sparsity structure of the Jacobian matrix for finite
            difference estimation, its shape must be (m, n). If the Jacobian has
            only few non-zero elements in *each* row, providing the sparsity
            structure will greatly speed up the computations [Curtis]_. A zero
            entry means that a corresponding element in the Jacobian is identically
            zero. If provided, forces the use of 'lsmr' trust-region solver.
            If None (default), then dense differencing will be used. Has no effect
            for 'lm' method.
        verbose : {0, 1, 2}, optional
            Level of algorithm's verbosity:
        
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations (not supported by 'lm'
              method).
        
        args, kwargs : tuple and dict, optional
            Additional arguments passed to `fun` and `jac`. Both empty by default.
            The calling signature is ``fun(x, *args, **kwargs)`` and the same for
            `jac`.
        
        Returns
        -------
        result : OptimizeResult
            `OptimizeResult` with the following fields defined:
        
            x : ndarray, shape (n,)
                Solution found.
            cost : float
                Value of the cost function at the solution.
            fun : ndarray, shape (m,)
                Vector of residuals at the solution.
            jac : ndarray, sparse matrix or LinearOperator, shape (m, n)
                Modified Jacobian matrix at the solution, in the sense that J^T J
                is a Gauss-Newton approximation of the Hessian of the cost function.
                The type is the same as the one used by the algorithm.
            grad : ndarray, shape (m,)
                Gradient of the cost function at the solution.
            optimality : float
                First-order optimality measure. In unconstrained problems, it is
                always the uniform norm of the gradient. In constrained problems,
                it is the quantity which was compared with `gtol` during iterations.
            active_mask : ndarray of int, shape (n,)
                Each component shows whether a corresponding constraint is active
                (that is, whether a variable is at the bound):
        
                *  0 : a constraint is not active.
                * -1 : a lower bound is active.
                *  1 : an upper bound is active.
        
                Might be somewhat arbitrary for 'trf' method as it generates a
                sequence of strictly feasible iterates and `active_mask` is
                determined within a tolerance threshold.
            nfev : int
                Number of function evaluations done. Methods 'trf' and 'dogbox' do
                not count function calls for numerical Jacobian approximation, as
                opposed to 'lm' method.
            njev : int or None
                Number of Jacobian evaluations done. If numerical Jacobian
                approximation is used in 'lm' method, it is set to None.
            status : int
                The reason for algorithm termination:
        
                * -1 : improper input parameters status returned from MINPACK.
                *  0 : the maximum number of function evaluations is exceeded.
                *  1 : `gtol` termination condition is satisfied.
                *  2 : `ftol` termination condition is satisfied.
                *  3 : `xtol` termination condition is satisfied.
                *  4 : Both `ftol` and `xtol` termination conditions are satisfied.
        
            message : str
                Verbal description of the termination reason.
            success : bool
                True if one of the convergence criteria is satisfied (`status` > 0).
        
        See Also
        --------
        leastsq : A legacy wrapper for the MINPACK implementation of the
                  Levenberg-Marquadt algorithm.
        curve_fit : Least-squares minimization applied to a curve-fitting problem.
        
        Notes
        -----
        Method 'lm' (Levenberg-Marquardt) calls a wrapper over least-squares
        algorithms implemented in MINPACK (lmder, lmdif). It runs the
        Levenberg-Marquardt algorithm formulated as a trust-region type algorithm.
        The implementation is based on paper [JJMore]_, it is very robust and
        efficient with a lot of smart tricks. It should be your first choice
        for unconstrained problems. Note that it doesn't support bounds. Also,
        it doesn't work when m < n.
        
        Method 'trf' (Trust Region Reflective) is motivated by the process of
        solving a system of equations, which constitute the first-order optimality
        condition for a bound-constrained minimization problem as formulated in
        [STIR]_. The algorithm iteratively solves trust-region subproblems
        augmented by a special diagonal quadratic term and with trust-region shape
        determined by the distance from the bounds and the direction of the
        gradient. This enhancements help to avoid making steps directly into bounds
        and efficiently explore the whole space of variables. To further improve
        convergence, the algorithm considers search directions reflected from the
        bounds. To obey theoretical requirements, the algorithm keeps iterates
        strictly feasible. With dense Jacobians trust-region subproblems are
        solved by an exact method very similar to the one described in [JJMore]_
        (and implemented in MINPACK). The difference from the MINPACK
        implementation is that a singular value decomposition of a Jacobian
        matrix is done once per iteration, instead of a QR decomposition and series
        of Givens rotation eliminations. For large sparse Jacobians a 2-D subspace
        approach of solving trust-region subproblems is used [STIR]_, [Byrd]_.
        The subspace is spanned by a scaled gradient and an approximate
        Gauss-Newton solution delivered by `scipy.sparse.linalg.lsmr`. When no
        constraints are imposed the algorithm is very similar to MINPACK and has
        generally comparable performance. The algorithm works quite robust in
        unbounded and bounded problems, thus it is chosen as a default algorithm.
        
        Method 'dogbox' operates in a trust-region framework, but considers
        rectangular trust regions as opposed to conventional ellipsoids [Voglis]_.
        The intersection of a current trust region and initial bounds is again
        rectangular, so on each iteration a quadratic minimization problem subject
        to bound constraints is solved approximately by Powell's dogleg method
        [NumOpt]_. The required Gauss-Newton step can be computed exactly for
        dense Jacobians or approximately by `scipy.sparse.linalg.lsmr` for large
        sparse Jacobians. The algorithm is likely to exhibit slow convergence when
        the rank of Jacobian is less than the number of variables. The algorithm
        often outperforms 'trf' in bounded problems with a small number of
        variables.
        
        Robust loss functions are implemented as described in [BA]_. The idea
        is to modify a residual vector and a Jacobian matrix on each iteration
        such that computed gradient and Gauss-Newton Hessian approximation match
        the true gradient and Hessian approximation of the cost function. Then
        the algorithm proceeds in a normal way, i.e., robust loss functions are
        implemented as a simple wrapper over standard least-squares algorithms.
        
        .. versionadded:: 0.17.0
        """
    def minimize(self, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        r"""Minimization of scalar function of one or more variables.
        
        Parameters
        ----------
        fun : callable
            The objective function to be minimized::
        
                fun(x, *args) -> float
        
            where ``x`` is a 1-D array with shape (n,) and ``args``
            is a tuple of the fixed parameters needed to completely
            specify the function.
        
            Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
            ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
            Rather than passing ``f0`` as the callable, wrap it to accept
            only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
            callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
            gathered before invoking this function.
        x0 : ndarray, shape (n,)
            Initial guess. Array of real elements of size (n,),
            where ``n`` is the number of independent variables.
        args : tuple, optional
            Extra arguments passed to the objective function and its
            derivatives (`fun`, `jac` and `hess` functions).
        method : str or callable, optional
            Type of solver.  Should be one of
        
            - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
            - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
            - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
            - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
            - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
            - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
            - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
            - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
            - 'COBYQA'      :ref:`(see here) <optimize.minimize-cobyqa>`
            - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
            - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
            - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
            - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
            - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
            - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
            - custom - a callable object, see below for description.
        
            If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
            depending on whether or not the problem has constraints or bounds.
        jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
            Method for computing the gradient vector. Only for CG, BFGS,
            Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov,
            trust-exact and trust-constr.
            If it is a callable, it should be a function that returns the gradient
            vector::
        
                jac(x, *args) -> array_like, shape (n,)
        
            where ``x`` is an array with shape (n,) and ``args`` is a tuple with
            the fixed parameters. If `jac` is a Boolean and is True, `fun` is
            assumed to return a tuple ``(f, g)`` containing the objective
            function and the gradient.
            Methods 'Newton-CG', 'trust-ncg', 'dogleg', 'trust-exact', and
            'trust-krylov' require that either a callable be supplied, or that
            `fun` return the objective and gradient.
            If None or False, the gradient will be estimated using 2-point finite
            difference estimation with an absolute step size.
            Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
            to select a finite difference scheme for numerical estimation of the
            gradient with a relative step size. These finite difference schemes
            obey any specified `bounds`.
        hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}, optional
            Method for computing the Hessian matrix. Only for Newton-CG, dogleg,
            trust-ncg, trust-krylov, trust-exact and trust-constr.
            If it is callable, it should return the Hessian matrix::
        
                hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)
        
            where ``x`` is a (n,) ndarray and ``args`` is a tuple with the fixed
            parameters.
            The keywords {'2-point', '3-point', 'cs'} can also be used to select
            a finite difference scheme for numerical estimation of the hessian.
            Alternatively, objects implementing the `HessianUpdateStrategy`
            interface can be used to approximate the Hessian. Available
            quasi-Newton methods implementing this interface are:
        
            - `BFGS`
            - `SR1`
        
            Not all of the options are available for each of the methods; for
            availability refer to the notes.
        hessp : callable, optional
            Hessian of objective function times an arbitrary vector p. Only for
            Newton-CG, trust-ncg, trust-krylov, trust-constr.
            Only one of `hessp` or `hess` needs to be given. If `hess` is
            provided, then `hessp` will be ignored. `hessp` must compute the
            Hessian times an arbitrary vector::
        
                hessp(x, p, *args) ->  ndarray shape (n,)
        
            where ``x`` is a (n,) ndarray, ``p`` is an arbitrary vector with
            dimension (n,) and ``args`` is a tuple with the fixed
            parameters.
        bounds : sequence or `Bounds`, optional
            Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell,
            trust-constr, COBYLA, and COBYQA methods. There are two ways to specify
            the bounds:
        
            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
        
        constraints : {Constraint, dict} or List of {Constraint, dict}, optional
            Constraints definition. Only for COBYLA, COBYQA, SLSQP and trust-constr.
        
            Constraints for 'trust-constr' and 'cobyqa' are defined as a single object
            or a list of objects specifying constraints to the optimization problem.
            Available constraints are:
        
            - `LinearConstraint`
            - `NonlinearConstraint`
        
            Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
            Each dictionary with fields:
        
            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.
        
            Equality constraint means that the constraint function result is to
            be zero whereas inequality means that it is to be non-negative.
            Note that COBYLA only supports inequality constraints.
        
        tol : float, optional
            Tolerance for termination. When `tol` is specified, the selected
            minimization algorithm sets some relevant solver-specific tolerance(s)
            equal to `tol`. For detailed control, use solver-specific
            options.
        options : dict, optional
            A dictionary of solver options. All methods except `TNC` accept the
            following generic options:
        
            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.
        
                For `TNC` use `maxfun` instead of `maxiter`.
            disp : bool
                Set to True to print convergence messages.
        
            For method-specific options, see :func:`show_options()`.
        callback : callable, optional
            A callable called after each iteration.
        
            All methods except TNC, SLSQP, and COBYLA support a callable with
            the signature::
        
                callback(intermediate_result: OptimizeResult)
        
            where ``intermediate_result`` is a keyword parameter containing an
            `OptimizeResult` with attributes ``x`` and ``fun``, the present values
            of the parameter vector and objective function. Note that the name
            of the parameter must be ``intermediate_result`` for the callback
            to be passed an `OptimizeResult`. These methods will also terminate if
            the callback raises ``StopIteration``.
        
            All methods except trust-constr (also) support a signature like::
        
                callback(xk)
        
            where ``xk`` is the current parameter vector.
        
            Introspection is used to determine which of the signatures above to
            invoke.
        
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.
        
        See also
        --------
        minimize_scalar : Interface to minimization algorithms for scalar
            univariate functions
        show_options : Additional options accepted by the solvers
        
        Notes
        -----
        This section describes the available solvers that can be selected by the
        'method' parameter. The default method is *BFGS*.
        
        **Unconstrained minimization**
        
        Method :ref:`CG <optimize.minimize-cg>` uses a nonlinear conjugate
        gradient algorithm by Polak and Ribiere, a variant of the
        Fletcher-Reeves method described in [5]_ pp.120-122. Only the
        first derivatives are used.
        
        Method :ref:`BFGS <optimize.minimize-bfgs>` uses the quasi-Newton
        method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS) [5]_
        pp. 136. It uses the first derivatives only. BFGS has proven good
        performance even for non-smooth optimizations. This method also
        returns an approximation of the Hessian inverse, stored as
        `hess_inv` in the OptimizeResult object.
        
        Method :ref:`Newton-CG <optimize.minimize-newtoncg>` uses a
        Newton-CG algorithm [5]_ pp. 168 (also known as the truncated
        Newton method). It uses a CG method to the compute the search
        direction. See also *TNC* method for a box-constrained
        minimization with a similar algorithm. Suitable for large-scale
        problems.
        
        Method :ref:`dogleg <optimize.minimize-dogleg>` uses the dog-leg
        trust-region algorithm [5]_ for unconstrained minimization. This
        algorithm requires the gradient and Hessian; furthermore the
        Hessian is required to be positive definite.
        
        Method :ref:`trust-ncg <optimize.minimize-trustncg>` uses the
        Newton conjugate gradient trust-region algorithm [5]_ for
        unconstrained minimization. This algorithm requires the gradient
        and either the Hessian or a function that computes the product of
        the Hessian with a given vector. Suitable for large-scale problems.
        
        Method :ref:`trust-krylov <optimize.minimize-trustkrylov>` uses
        the Newton GLTR trust-region algorithm [14]_, [15]_ for unconstrained
        minimization. This algorithm requires the gradient
        and either the Hessian or a function that computes the product of
        the Hessian with a given vector. Suitable for large-scale problems.
        On indefinite problems it requires usually less iterations than the
        `trust-ncg` method and is recommended for medium and large-scale problems.
        
        Method :ref:`trust-exact <optimize.minimize-trustexact>`
        is a trust-region method for unconstrained minimization in which
        quadratic subproblems are solved almost exactly [13]_. This
        algorithm requires the gradient and the Hessian (which is
        *not* required to be positive definite). It is, in many
        situations, the Newton method to converge in fewer iterations
        and the most recommended for small and medium-size problems.
        
        **Bound-Constrained minimization**
        
        Method :ref:`Nelder-Mead <optimize.minimize-neldermead>` uses the
        Simplex algorithm [1]_, [2]_. This algorithm is robust in many
        applications. However, if numerical computation of derivative can be
        trusted, other algorithms using the first and/or second derivatives
        information might be preferred for their better performance in
        general.
        
        Method :ref:`L-BFGS-B <optimize.minimize-lbfgsb>` uses the L-BFGS-B
        algorithm [6]_, [7]_ for bound constrained minimization.
        
        Method :ref:`Powell <optimize.minimize-powell>` is a modification
        of Powell's method [3]_, [4]_ which is a conjugate direction
        method. It performs sequential one-dimensional minimizations along
        each vector of the directions set (`direc` field in `options` and
        `info`), which is updated at each iteration of the main
        minimization loop. The function need not be differentiable, and no
        derivatives are taken. If bounds are not provided, then an
        unbounded line search will be used. If bounds are provided and
        the initial guess is within the bounds, then every function
        evaluation throughout the minimization procedure will be within
        the bounds. If bounds are provided, the initial guess is outside
        the bounds, and `direc` is full rank (default has full rank), then
        some function evaluations during the first iteration may be
        outside the bounds, but every function evaluation after the first
        iteration will be within the bounds. If `direc` is not full rank,
        then some parameters may not be optimized and the solution is not
        guaranteed to be within the bounds.
        
        Method :ref:`TNC <optimize.minimize-tnc>` uses a truncated Newton
        algorithm [5]_, [8]_ to minimize a function with variables subject
        to bounds. This algorithm uses gradient information; it is also
        called Newton Conjugate-Gradient. It differs from the *Newton-CG*
        method described above as it wraps a C implementation and allows
        each variable to be given upper and lower bounds.
        
        **Constrained Minimization**
        
        Method :ref:`COBYLA <optimize.minimize-cobyla>` uses the
        Constrained Optimization BY Linear Approximation (COBYLA) method
        [9]_, [10]_, [11]_. The algorithm is based on linear
        approximations to the objective function and each constraint. The
        method wraps a FORTRAN implementation of the algorithm. The
        constraints functions 'fun' may return either a single number
        or an array or list of numbers.
        
        Method :ref:`COBYQA <optimize.minimize-cobyqa>` uses the Constrained
        Optimization BY Quadratic Approximations (COBYQA) method [18]_. The
        algorithm is a derivative-free trust-region SQP method based on quadratic
        approximations to the objective function and each nonlinear constraint. The
        bounds are treated as unrelaxable constraints, in the sense that the
        algorithm always respects them throughout the optimization process.
        
        Method :ref:`SLSQP <optimize.minimize-slsqp>` uses Sequential
        Least SQuares Programming to minimize a function of several
        variables with any combination of bounds, equality and inequality
        constraints. The method wraps the SLSQP Optimization subroutine
        originally implemented by Dieter Kraft [12]_. Note that the
        wrapper handles infinite values in bounds by converting them into
        large floating values.
        
        Method :ref:`trust-constr <optimize.minimize-trustconstr>` is a
        trust-region algorithm for constrained optimization. It switches
        between two implementations depending on the problem definition.
        It is the most versatile constrained minimization algorithm
        implemented in SciPy and the most appropriate for large-scale problems.
        For equality constrained problems it is an implementation of Byrd-Omojokun
        Trust-Region SQP method described in [17]_ and in [5]_, p. 549. When
        inequality constraints are imposed as well, it switches to the trust-region
        interior point method described in [16]_. This interior point algorithm,
        in turn, solves inequality constraints by introducing slack variables
        and solving a sequence of equality-constrained barrier problems
        for progressively smaller values of the barrier parameter.
        The previously described equality constrained SQP method is
        used to solve the subproblems with increasing levels of accuracy
        as the iterate gets closer to a solution.
        
        **Finite-Difference Options**
        
        For Method :ref:`trust-constr <optimize.minimize-trustconstr>`
        the gradient and the Hessian may be approximated using
        three finite-difference schemes: {'2-point', '3-point', 'cs'}.
        The scheme 'cs' is, potentially, the most accurate but it
        requires the function to correctly handle complex inputs and to
        be differentiable in the complex plane. The scheme '3-point' is more
        accurate than '2-point' but requires twice as many operations. If the
        gradient is estimated via finite-differences the Hessian must be
        estimated using one of the quasi-Newton strategies.
        
        **Method specific options for the** `hess` **keyword**
        
        +--------------+------+----------+-------------------------+-----+
        | method/Hess  | None | callable | '2-point/'3-point'/'cs' | HUS |
        +==============+======+==========+=========================+=====+
        | Newton-CG    | x    | (n, n)   | x                       | x   |
        |              |      | LO       |                         |     |
        +--------------+------+----------+-------------------------+-----+
        | dogleg       |      | (n, n)   |                         |     |
        +--------------+------+----------+-------------------------+-----+
        | trust-ncg    |      | (n, n)   | x                       | x   |
        +--------------+------+----------+-------------------------+-----+
        | trust-krylov |      | (n, n)   | x                       | x   |
        +--------------+------+----------+-------------------------+-----+
        | trust-exact  |      | (n, n)   |                         |     |
        +--------------+------+----------+-------------------------+-----+
        | trust-constr | x    | (n, n)   |  x                      | x   |
        |              |      | LO       |                         |     |
        |              |      | sp       |                         |     |
        +--------------+------+----------+-------------------------+-----+
        
        where LO=LinearOperator, sp=Sparse matrix, HUS=HessianUpdateStrategy
        
        **Custom minimizers**
        
        It may be useful to pass a custom minimization method, for example
        when using a frontend to this method such as `scipy.optimize.basinhopping`
        or a different library.  You can simply pass a callable as the ``method``
        parameter.
        
        The callable is called as ``method(fun, x0, args, **kwargs, **options)``
        where ``kwargs`` corresponds to any other parameters passed to `minimize`
        (such as `callback`, `hess`, etc.), except the `options` dict, which has
        its contents also passed as `method` parameters pair by pair.  Also, if
        `jac` has been passed as a bool type, `jac` and `fun` are mangled so that
        `fun` returns just the function values and `jac` is converted to a function
        returning the Jacobian.  The method shall return an `OptimizeResult`
        object.
        
        The provided `method` callable must be able to accept (and possibly ignore)
        arbitrary parameters; the set of parameters accepted by `minimize` may
        expand in future versions and then these parameters will be passed to
        the method.  You can find an example in the scipy.optimize tutorial.
        """
    def minimize_scalar(self, bracket=None, bounds=None, args=(), method=None, tol=None, options=None):
        r"""Local minimization of scalar function of one variable.
        
        Parameters
        ----------
        fun : callable
            Objective function.
            Scalar function, must return a scalar.
        
            Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
            ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
            Rather than passing ``f0`` as the callable, wrap it to accept
            only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
            callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
            gathered before invoking this function.
        
        bracket : sequence, optional
            For methods 'brent' and 'golden', `bracket` defines the bracketing
            interval and is required.
            Either a triple ``(xa, xb, xc)`` satisfying ``xa < xb < xc`` and
            ``func(xb) < func(xa) and  func(xb) < func(xc)``, or a pair
            ``(xa, xb)`` to be used as initial points for a downhill bracket search
            (see `scipy.optimize.bracket`).
            The minimizer ``res.x`` will not necessarily satisfy
            ``xa <= res.x <= xb``.
        bounds : sequence, optional
            For method 'bounded', `bounds` is mandatory and must have two finite
            items corresponding to the optimization bounds.
        args : tuple, optional
            Extra arguments passed to the objective function.
        method : str or callable, optional
            Type of solver.  Should be one of:
        
            - :ref:`Brent <optimize.minimize_scalar-brent>`
            - :ref:`Bounded <optimize.minimize_scalar-bounded>`
            - :ref:`Golden <optimize.minimize_scalar-golden>`
            - custom - a callable object (added in version 0.14.0), see below
        
            Default is "Bounded" if bounds are provided and "Brent" otherwise.
            See the 'Notes' section for details of each solver.
        
        tol : float, optional
            Tolerance for termination. For detailed control, use solver-specific
            options.
        options : dict, optional
            A dictionary of solver options.
        
            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.
        
            See :func:`show_options()` for solver-specific options.
        
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.
        
        See also
        --------
        minimize : Interface to minimization algorithms for scalar multivariate
            functions
        show_options : Additional options accepted by the solvers
        
        Notes
        -----
        This section describes the available solvers that can be selected by the
        'method' parameter. The default method is the ``"Bounded"`` Brent method if
        `bounds` are passed and unbounded ``"Brent"`` otherwise.
        
        Method :ref:`Brent <optimize.minimize_scalar-brent>` uses Brent's
        algorithm [1]_ to find a local minimum.  The algorithm uses inverse
        parabolic interpolation when possible to speed up convergence of
        the golden section method.
        
        Method :ref:`Golden <optimize.minimize_scalar-golden>` uses the
        golden section search technique [1]_. It uses analog of the bisection
        method to decrease the bracketed interval. It is usually
        preferable to use the *Brent* method.
        
        Method :ref:`Bounded <optimize.minimize_scalar-bounded>` can
        perform bounded minimization [2]_ [3]_. It uses the Brent method to find a
        local minimum in the interval x1 < xopt < x2.
        
        Note that the Brent and Golden methods do not guarantee success unless a
        valid ``bracket`` triple is provided. If a three-point bracket cannot be
        found, consider `scipy.optimize.minimize`. Also, all methods are intended
        only for local minimization. When the function of interest has more than
        one local minimum, consider :ref:`global_optimization`.
        
        **Custom minimizers**
        
        It may be useful to pass a custom minimization method, for example
        when using some library frontend to minimize_scalar. You can simply
        pass a callable as the ``method`` parameter.
        
        The callable is called as ``method(fun, args, **kwargs, **options)``
        where ``kwargs`` corresponds to any other parameters passed to `minimize`
        (such as `bracket`, `tol`, etc.), except the `options` dict, which has
        its contents also passed as `method` parameters pair by pair.  The method
        shall return an `OptimizeResult` object.
        
        The provided `method` callable must be able to accept (and possibly ignore)
        arbitrary parameters; the set of parameters accepted by `minimize` may
        expand in future versions and then these parameters will be passed to
        the method. You can find an example in the scipy.optimize tutorial.
        
        .. versionadded:: 0.11.0
        """
    def newton(self, x0, fprime=None, args=(), tol=1.48e-08, maxiter=50, fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True):
        r"""
        Find a root of a real or complex function using the Newton-Raphson
        (or secant or Halley's) method.
        
        Find a root of the scalar-valued function `func` given a nearby scalar
        starting point `x0`.
        The Newton-Raphson method is used if the derivative `fprime` of `func`
        is provided, otherwise the secant method is used. If the second order
        derivative `fprime2` of `func` is also provided, then Halley's method is
        used.
        
        If `x0` is a sequence with more than one item, `newton` returns an array:
        the roots of the function from each (scalar) starting point in `x0`.
        In this case, `func` must be vectorized to return a sequence or array of
        the same shape as its first argument. If `fprime` (`fprime2`) is given,
        then its return must also have the same shape: each element is the first
        (second) derivative of `func` with respect to its only variable evaluated
        at each element of its first argument.
        
        `newton` is for finding roots of a scalar-valued functions of a single
        variable. For problems involving several variables, see `root`.
        
        Parameters
        ----------
        func : callable
            The function whose root is wanted. It must be a function of a
            single variable of the form ``f(x,a,b,c...)``, where ``a,b,c...``
            are extra arguments that can be passed in the `args` parameter.
        x0 : float, sequence, or ndarray
            An initial estimate of the root that should be somewhere near the
            actual root. If not scalar, then `func` must be vectorized and return
            a sequence or array of the same shape as its first argument.
        fprime : callable, optional
            The derivative of the function when available and convenient. If it
            is None (default), then the secant method is used.
        args : tuple, optional
            Extra arguments to be used in the function call.
        tol : float, optional
            The allowable error of the root's value. If `func` is complex-valued,
            a larger `tol` is recommended as both the real and imaginary parts
            of `x` contribute to ``|x - x0|``.
        maxiter : int, optional
            Maximum number of iterations.
        fprime2 : callable, optional
            The second order derivative of the function when available and
            convenient. If it is None (default), then the normal Newton-Raphson
            or the secant method is used. If it is not None, then Halley's method
            is used.
        x1 : float, optional
            Another estimate of the root that should be somewhere near the
            actual root. Used if `fprime` is not provided.
        rtol : float, optional
            Tolerance (relative) for termination.
        full_output : bool, optional
            If `full_output` is False (default), the root is returned.
            If True and `x0` is scalar, the return value is ``(x, r)``, where ``x``
            is the root and ``r`` is a `RootResults` object.
            If True and `x0` is non-scalar, the return value is ``(x, converged,
            zero_der)`` (see Returns section for details).
        disp : bool, optional
            If True, raise a RuntimeError if the algorithm didn't converge, with
            the error message containing the number of iterations and current
            function value. Otherwise, the convergence status is recorded in a
            `RootResults` return object.
            Ignored if `x0` is not scalar.
            *Note: this has little to do with displaying, however,
            the `disp` keyword cannot be renamed for backwards compatibility.*
        
        Returns
        -------
        root : float, sequence, or ndarray
            Estimated location where function is zero.
        r : `RootResults`, optional
            Present if ``full_output=True`` and `x0` is scalar.
            Object containing information about the convergence. In particular,
            ``r.converged`` is True if the routine converged.
        converged : ndarray of bool, optional
            Present if ``full_output=True`` and `x0` is non-scalar.
            For vector functions, indicates which elements converged successfully.
        zero_der : ndarray of bool, optional
            Present if ``full_output=True`` and `x0` is non-scalar.
            For vector functions, indicates which elements had a zero derivative.
        
        See Also
        --------
        root_scalar : interface to root solvers for scalar functions
        root : interface to root solvers for multi-input, multi-output functions
        
        Notes
        -----
        The convergence rate of the Newton-Raphson method is quadratic,
        the Halley method is cubic, and the secant method is
        sub-quadratic. This means that if the function is well-behaved
        the actual error in the estimated root after the nth iteration
        is approximately the square (cube for Halley) of the error
        after the (n-1)th step. However, the stopping criterion used
        here is the step size and there is no guarantee that a root
        has been found. Consequently, the result should be verified.
        Safer algorithms are brentq, brenth, ridder, and bisect,
        but they all require that the root first be bracketed in an
        interval where the function changes sign. The brentq algorithm
        is recommended for general use in one dimensional problems
        when such an interval has been found.
        
        When `newton` is used with arrays, it is best suited for the following
        types of problems:
        
        * The initial guesses, `x0`, are all relatively the same distance from
          the roots.
        * Some or all of the extra arguments, `args`, are also arrays so that a
          class of similar problems can be solved together.
        * The size of the initial guesses, `x0`, is larger than O(100) elements.
          Otherwise, a naive loop may perform as well or better than a vector.
        """
    def nquad(self, ranges, args=None, opts=None, full_output=False):
        r"""
        Integration over multiple variables.
        
        Wraps `quad` to enable integration over multiple variables.
        Various options allow improved integration of discontinuous functions, as
        well as the use of weighted integration, and generally finer control of the
        integration process.
        
        Parameters
        ----------
        func : {callable, scipy.LowLevelCallable}
            The function to be integrated. Has arguments of ``x0, ... xn``,
            ``t0, ... tm``, where integration is carried out over ``x0, ... xn``,
            which must be floats.  Where ``t0, ... tm`` are extra arguments
            passed in args.
            Function signature should be ``func(x0, x1, ..., xn, t0, t1, ..., tm)``.
            Integration is carried out in order.  That is, integration over ``x0``
            is the innermost integral, and ``xn`` is the outermost.
        
            If the user desires improved integration performance, then `f` may
            be a `scipy.LowLevelCallable` with one of the signatures::
        
                double func(int n, double *xx)
                double func(int n, double *xx, void *user_data)
        
            where ``n`` is the number of variables and args.  The ``xx`` array
            contains the coordinates and extra arguments. ``user_data`` is the data
            contained in the `scipy.LowLevelCallable`.
        ranges : iterable object
            Each element of ranges may be either a sequence  of 2 numbers, or else
            a callable that returns such a sequence. ``ranges[0]`` corresponds to
            integration over x0, and so on. If an element of ranges is a callable,
            then it will be called with all of the integration arguments available,
            as well as any parametric arguments. e.g., if
            ``func = f(x0, x1, x2, t0, t1)``, then ``ranges[0]`` may be defined as
            either ``(a, b)`` or else as ``(a, b) = range0(x1, x2, t0, t1)``.
        args : iterable object, optional
            Additional arguments ``t0, ... tn``, required by ``func``, ``ranges``,
            and ``opts``.
        opts : iterable object or dict, optional
            Options to be passed to `quad`. May be empty, a dict, or
            a sequence of dicts or functions that return a dict. If empty, the
            default options from scipy.integrate.quad are used. If a dict, the same
            options are used for all levels of integraion. If a sequence, then each
            element of the sequence corresponds to a particular integration. e.g.,
            ``opts[0]`` corresponds to integration over ``x0``, and so on. If a
            callable, the signature must be the same as for ``ranges``. The
            available options together with their default values are:
        
              - epsabs = 1.49e-08
              - epsrel = 1.49e-08
              - limit  = 50
              - points = None
              - weight = None
              - wvar   = None
              - wopts  = None
        
            For more information on these options, see `quad`.
        
        full_output : bool, optional
            Partial implementation of ``full_output`` from scipy.integrate.quad.
            The number of integrand function evaluations ``neval`` can be obtained
            by setting ``full_output=True`` when calling nquad.
        
        Returns
        -------
        result : float
            The result of the integration.
        abserr : float
            The maximum of the estimates of the absolute error in the various
            integration results.
        out_dict : dict, optional
            A dict containing additional information on the integration.
        
        See Also
        --------
        quad : 1-D numerical integration
        dblquad, tplquad : double and triple integrals
        fixed_quad : fixed-order Gaussian quadrature
        
        Notes
        -----
        For valid results, the integral must converge; behavior for divergent
        integrals is not guaranteed.
        
        **Details of QUADPACK level routines**
        
        `nquad` calls routines from the FORTRAN library QUADPACK. This section
        provides details on the conditions for each routine to be called and a
        short description of each routine. The routine called depends on
        `weight`, `points` and the integration limits `a` and `b`.
        
        ================  ==============  ==========  =====================
        QUADPACK routine  `weight`        `points`    infinite bounds
        ================  ==============  ==========  =====================
        qagse             None            No          No
        qagie             None            No          Yes
        qagpe             None            Yes         No
        qawoe             'sin', 'cos'    No          No
        qawfe             'sin', 'cos'    No          either `a` or `b`
        qawse             'alg*'          No          No
        qawce             'cauchy'        No          No
        ================  ==============  ==========  =====================
        
        The following provides a short description from [1]_ for each
        routine.
        
        qagse
            is an integrator based on globally adaptive interval
            subdivision in connection with extrapolation, which will
            eliminate the effects of integrand singularities of
            several types.
        qagie
            handles integration over infinite intervals. The infinite range is
            mapped onto a finite interval and subsequently the same strategy as
            in ``QAGS`` is applied.
        qagpe
            serves the same purposes as QAGS, but also allows the
            user to provide explicit information about the location
            and type of trouble-spots i.e. the abscissae of internal
            singularities, discontinuities and other difficulties of
            the integrand function.
        qawoe
            is an integrator for the evaluation of
            :math:`\int^b_a \cos(\omega x)f(x)dx` or
            :math:`\int^b_a \sin(\omega x)f(x)dx`
            over a finite interval [a,b], where :math:`\omega` and :math:`f`
            are specified by the user. The rule evaluation component is based
            on the modified Clenshaw-Curtis technique
        
            An adaptive subdivision scheme is used in connection
            with an extrapolation procedure, which is a modification
            of that in ``QAGS`` and allows the algorithm to deal with
            singularities in :math:`f(x)`.
        qawfe
            calculates the Fourier transform
            :math:`\int^\infty_a \cos(\omega x)f(x)dx` or
            :math:`\int^\infty_a \sin(\omega x)f(x)dx`
            for user-provided :math:`\omega` and :math:`f`. The procedure of
            ``QAWO`` is applied on successive finite intervals, and convergence
            acceleration by means of the :math:`\varepsilon`-algorithm is applied
            to the series of integral approximations.
        qawse
            approximate :math:`\int^b_a w(x)f(x)dx`, with :math:`a < b` where
            :math:`w(x) = (x-a)^{\alpha}(b-x)^{\beta}v(x)` with
            :math:`\alpha,\beta > -1`, where :math:`v(x)` may be one of the
            following functions: :math:`1`, :math:`\log(x-a)`, :math:`\log(b-x)`,
            :math:`\log(x-a)\log(b-x)`.
        
            The user specifies :math:`\alpha`, :math:`\beta` and the type of the
            function :math:`v`. A globally adaptive subdivision strategy is
            applied, with modified Clenshaw-Curtis integration on those
            subintervals which contain `a` or `b`.
        qawce
            compute :math:`\int^b_a f(x) / (x-c)dx` where the integral must be
            interpreted as a Cauchy principal value integral, for user specified
            :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
            Clenshaw-Curtis integration is used on those intervals containing the
            point :math:`x = c`.
        """
    def nsum(self, a, b, step=1, args=(), log=False, maxterms=1048576, tolerances=None):
        r"""Evaluate a convergent finite or infinite series.
        
        For finite `a` and `b`, this evaluates::
        
            f(a + np.arange(n)*step).sum()
        
        where ``n = int((b - a) / step) + 1``, where `f` is smooth, positive, and
        unimodal. The number of terms in the sum may be very large or infinite,
        in which case a partial sum is evaluated directly and the remainder is
        approximated using integration.
        
        Parameters
        ----------
        f : callable
            The function that evaluates terms to be summed. The signature must be::
        
                f(x: ndarray, *args) -> ndarray
        
            where each element of ``x`` is a finite real and ``args`` is a tuple,
            which may contain an arbitrary number of arrays that are broadcastable
            with ``x``.
        
            `f` must be an elementwise function: each element ``f(x)[i]``
            must equal ``f(x[i])`` for all indices ``i``. It must not mutate the
            array ``x`` or the arrays in ``args``, and it must return NaN where
            the argument is NaN.
        
            `f` must represent a smooth, positive, unimodal function of `x` defined at
            *all reals* between `a` and `b`.
        a, b : float array_like
            Real lower and upper limits of summed terms. Must be broadcastable.
            Each element of `a` must be less than the corresponding element in `b`.
        step : float array_like
            Finite, positive, real step between summed terms. Must be broadcastable
            with `a` and `b`. Note that the number of terms included in the sum will
            be ``floor((b - a) / step)`` + 1; adjust `b` accordingly to ensure
            that ``f(b)`` is included if intended.
        args : tuple of array_like, optional
            Additional positional arguments to be passed to `f`. Must be arrays
            broadcastable with `a`, `b`, and `step`. If the callable to be summed
            requires arguments that are not broadcastable with `a`, `b`, and `step`,
            wrap that callable with `f` such that `f` accepts only `x` and
            broadcastable ``*args``. See Examples.
        log : bool, default: False
            Setting to True indicates that `f` returns the log of the terms
            and that `atol` and `rtol` are expressed as the logs of the absolute
            and relative errors. In this case, the result object will contain the
            log of the sum and error. This is useful for summands for which
            numerical underflow or overflow would lead to inaccuracies.
        maxterms : int, default: 2**20
            The maximum number of terms to evaluate for direct summation.
            Additional function evaluations may be performed for input
            validation and integral evaluation.
        atol, rtol : float, optional
            Absolute termination tolerance (default: 0) and relative termination
            tolerance (default: ``eps**0.5``, where ``eps`` is the precision of
            the result dtype), respectively. Must be non-negative
            and finite if `log` is False, and must be expressed as the log of a
            non-negative and finite number if `log` is True.
        
        Returns
        -------
        res : _RichResult
            An object similar to an instance of `scipy.optimize.OptimizeResult` with the
            following attributes. (The descriptions are written as though the values will
            be scalars; however, if `f` returns an array, the outputs will be
            arrays of the same shape.)
        
            success : bool
                ``True`` when the algorithm terminated successfully (status ``0``);
                ``False`` otherwise.
            status : int array
                An integer representing the exit status of the algorithm.
        
                - ``0`` : The algorithm converged to the specified tolerances.
                - ``-1`` : Element(s) of `a`, `b`, or `step` are invalid
                - ``-2`` : Numerical integration reached its iteration limit;
                  the sum may be divergent.
                - ``-3`` : A non-finite value was encountered.
                - ``-4`` : The magnitude of the last term of the partial sum exceeds
                  the tolerances, so the error estimate exceeds the tolerances.
                  Consider increasing `maxterms` or loosening `tolerances`.
                  Alternatively, the callable may not be unimodal, or the limits of
                  summation may be too far from the function maximum. Consider
                  increasing `maxterms` or breaking the sum into pieces.
        
            sum : float array
                An estimate of the sum.
            error : float array
                An estimate of the absolute error, assuming all terms are non-negative,
                the function is computed exactly, and direct summation is accurate to
                the precision of the result dtype.
            nfev : int array
                The number of points at which `f` was evaluated.
        
        See Also
        --------
        mpmath.nsum
        
        Notes
        -----
        The method implemented for infinite summation is related to the integral
        test for convergence of an infinite series: assuming `step` size 1 for
        simplicity of exposition, the sum of a monotone decreasing function is bounded by
        
        .. math::
        
            \int_u^\infty f(x) dx \leq \sum_{k=u}^\infty f(k) \leq \int_u^\infty f(x) dx + f(u)
        
        Let :math:`a` represent  `a`, :math:`n` represent `maxterms`, :math:`\epsilon_a`
        represent `atol`, and :math:`\epsilon_r` represent `rtol`.
        The implementation first evaluates the integral :math:`S_l=\int_a^\infty f(x) dx`
        as a lower bound of the infinite sum. Then, it seeks a value :math:`c > a` such
        that :math:`f(c) < \epsilon_a + S_l \epsilon_r`, if it exists; otherwise,
        let :math:`c = a + n`. Then the infinite sum is approximated as
        
        .. math::
        
            \sum_{k=a}^{c-1} f(k) + \int_c^\infty f(x) dx + f(c)/2
        
        and the reported error is :math:`f(c)/2` plus the error estimate of
        numerical integration. Note that the integral approximations may require
        evaluation of the function at points besides those that appear in the sum,
        so `f` must be a continuous and monotonically decreasing function defined
        for all reals within the integration interval. However, due to the nature
        of the integral approximation, the shape of the function between points
        that appear in the sum has little effect. If there is not a natural
        extension of the function to all reals, consider using linear interpolation,
        which is easy to evaluate and preserves monotonicity.
        
        The approach described above is generalized for non-unit
        `step` and finite `b` that is too large for direct evaluation of the sum,
        i.e. ``b - a + 1 > maxterms``. It is further generalized to unimodal
        functions by directly summing terms surrounding the maximum.
        This strategy may fail:
        
        - If the left limit is finite and the maximum is far from it.
        - If the right limit is finite and the maximum is far from it.
        - If both limits are finite and the maximum is far from the origin.
        
        In these cases, accuracy may be poor, and `nsum` may return status code ``4``.
        
        Although the callable `f` must be non-negative and unimodal,
        `nsum` can be used to evaluate more general forms of series. For instance, to
        evaluate an alternating series, pass a callable that returns the difference
        between pairs of adjacent terms, and adjust `step` accordingly. See Examples.
        """
    def qmc_quad(self, a, b, n_estimates=8, n_points=1024, qrng=None, log=False):
        r"""
        Compute an integral in N-dimensions using Quasi-Monte Carlo quadrature.
        
        Parameters
        ----------
        func : callable
            The integrand. Must accept a single argument ``x``, an array which
            specifies the point(s) at which to evaluate the scalar-valued
            integrand, and return the value(s) of the integrand.
            For efficiency, the function should be vectorized to accept an array of
            shape ``(d, n_points)``, where ``d`` is the number of variables (i.e.
            the dimensionality of the function domain) and `n_points` is the number
            of quadrature points, and return an array of shape ``(n_points,)``,
            the integrand at each quadrature point.
        a, b : array-like
            One-dimensional arrays specifying the lower and upper integration
            limits, respectively, of each of the ``d`` variables.
        n_estimates, n_points : int, optional
            `n_estimates` (default: 8) statistically independent QMC samples, each
            of `n_points` (default: 1024) points, will be generated by `qrng`.
            The total number of points at which the integrand `func` will be
            evaluated is ``n_points * n_estimates``. See Notes for details.
        qrng : `~scipy.stats.qmc.QMCEngine`, optional
            An instance of the QMCEngine from which to sample QMC points.
            The QMCEngine must be initialized to a number of dimensions ``d``
            corresponding with the number of variables ``x1, ..., xd`` passed to
            `func`.
            The provided QMCEngine is used to produce the first integral estimate.
            If `n_estimates` is greater than one, additional QMCEngines are
            spawned from the first (with scrambling enabled, if it is an option.)
            If a QMCEngine is not provided, the default `scipy.stats.qmc.Halton`
            will be initialized with the number of dimensions determine from
            the length of `a`.
        log : boolean, default: False
            When set to True, `func` returns the log of the integrand, and
            the result object contains the log of the integral.
        
        Returns
        -------
        result : object
            A result object with attributes:
        
            integral : float
                The estimate of the integral.
            standard_error :
                The error estimate. See Notes for interpretation.
        
        Notes
        -----
        Values of the integrand at each of the `n_points` points of a QMC sample
        are used to produce an estimate of the integral. This estimate is drawn
        from a population of possible estimates of the integral, the value of
        which we obtain depends on the particular points at which the integral
        was evaluated. We perform this process `n_estimates` times, each time
        evaluating the integrand at different scrambled QMC points, effectively
        drawing i.i.d. random samples from the population of integral estimates.
        The sample mean :math:`m` of these integral estimates is an
        unbiased estimator of the true value of the integral, and the standard
        error of the mean :math:`s` of these estimates may be used to generate
        confidence intervals using the t distribution with ``n_estimates - 1``
        degrees of freedom. Perhaps counter-intuitively, increasing `n_points`
        while keeping the total number of function evaluation points
        ``n_points * n_estimates`` fixed tends to reduce the actual error, whereas
        increasing `n_estimates` tends to decrease the error estimate.
        """
    def quad(self, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False):
        r"""
        Compute a definite integral.
        
        Integrate func from `a` to `b` (possibly infinite interval) using a
        technique from the Fortran library QUADPACK.
        
        Parameters
        ----------
        func : {function, scipy.LowLevelCallable}
            A Python function or method to integrate. If `func` takes many
            arguments, it is integrated along the axis corresponding to the
            first argument.
        
            If the user desires improved integration performance, then `f` may
            be a `scipy.LowLevelCallable` with one of the signatures::
        
                double func(double x)
                double func(double x, void *user_data)
                double func(int n, double *xx)
                double func(int n, double *xx, void *user_data)
        
            The ``user_data`` is the data contained in the `scipy.LowLevelCallable`.
            In the call forms with ``xx``,  ``n`` is the length of the ``xx``
            array which contains ``xx[0] == x`` and the rest of the items are
            numbers contained in the ``args`` argument of quad.
        
            In addition, certain ctypes call signatures are supported for
            backward compatibility, but those should not be used in new code.
        a : float
            Lower limit of integration (use -numpy.inf for -infinity).
        b : float
            Upper limit of integration (use numpy.inf for +infinity).
        args : tuple, optional
            Extra arguments to pass to `func`.
        full_output : int, optional
            Non-zero to return a dictionary of integration information.
            If non-zero, warning messages are also suppressed and the
            message is appended to the output tuple.
        complex_func : bool, optional
            Indicate if the function's (`func`) return type is real
            (``complex_func=False``: default) or complex (``complex_func=True``).
            In both cases, the function's argument is real.
            If full_output is also non-zero, the `infodict`, `message`, and
            `explain` for the real and complex components are returned in
            a dictionary with keys "real output" and "imag output".
        
        Returns
        -------
        y : float
            The integral of func from `a` to `b`.
        abserr : float
            An estimate of the absolute error in the result.
        infodict : dict
            A dictionary containing additional information.
        message
            A convergence message.
        explain
            Appended only with 'cos' or 'sin' weighting and infinite
            integration limits, it contains an explanation of the codes in
            infodict['ierlst']
        
        Other Parameters
        ----------------
        epsabs : float or int, optional
            Absolute error tolerance. Default is 1.49e-8. `quad` tries to obtain
            an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
            where ``i`` = integral of `func` from `a` to `b`, and ``result`` is the
            numerical approximation. See `epsrel` below.
        epsrel : float or int, optional
            Relative error tolerance. Default is 1.49e-8.
            If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
            and ``50 * (machine epsilon)``. See `epsabs` above.
        limit : float or int, optional
            An upper bound on the number of subintervals used in the adaptive
            algorithm.
        points : (sequence of floats,ints), optional
            A sequence of break points in the bounded integration interval
            where local difficulties of the integrand may occur (e.g.,
            singularities, discontinuities). The sequence does not have
            to be sorted. Note that this option cannot be used in conjunction
            with ``weight``.
        weight : float or int, optional
            String indicating weighting function. Full explanation for this
            and the remaining arguments can be found below.
        wvar : optional
            Variables for use with weighting functions.
        wopts : optional
            Optional input for reusing Chebyshev moments.
        maxp1 : float or int, optional
            An upper bound on the number of Chebyshev moments.
        limlst : int, optional
            Upper bound on the number of cycles (>=3) for use with a sinusoidal
            weighting and an infinite end-point.
        
        See Also
        --------
        dblquad : double integral
        tplquad : triple integral
        nquad : n-dimensional integrals (uses `quad` recursively)
        fixed_quad : fixed-order Gaussian quadrature
        simpson : integrator for sampled data
        romb : integrator for sampled data
        scipy.special : for coefficients and roots of orthogonal polynomials
        
        Notes
        -----
        For valid results, the integral must converge; behavior for divergent
        integrals is not guaranteed.
        
        **Extra information for quad() inputs and outputs**
        
        If full_output is non-zero, then the third output argument
        (infodict) is a dictionary with entries as tabulated below. For
        infinite limits, the range is transformed to (0,1) and the
        optional outputs are given with respect to this transformed range.
        Let M be the input argument limit and let K be infodict['last'].
        The entries are:
        
        'neval'
            The number of function evaluations.
        'last'
            The number, K, of subintervals produced in the subdivision process.
        'alist'
            A rank-1 array of length M, the first K elements of which are the
            left end points of the subintervals in the partition of the
            integration range.
        'blist'
            A rank-1 array of length M, the first K elements of which are the
            right end points of the subintervals.
        'rlist'
            A rank-1 array of length M, the first K elements of which are the
            integral approximations on the subintervals.
        'elist'
            A rank-1 array of length M, the first K elements of which are the
            moduli of the absolute error estimates on the subintervals.
        'iord'
            A rank-1 integer array of length M, the first L elements of
            which are pointers to the error estimates over the subintervals
            with ``L=K`` if ``K<=M/2+2`` or ``L=M+1-K`` otherwise. Let I be the
            sequence ``infodict['iord']`` and let E be the sequence
            ``infodict['elist']``.  Then ``E[I[1]], ..., E[I[L]]`` forms a
            decreasing sequence.
        
        If the input argument points is provided (i.e., it is not None),
        the following additional outputs are placed in the output
        dictionary. Assume the points sequence is of length P.
        
        'pts'
            A rank-1 array of length P+2 containing the integration limits
            and the break points of the intervals in ascending order.
            This is an array giving the subintervals over which integration
            will occur.
        'level'
            A rank-1 integer array of length M (=limit), containing the
            subdivision levels of the subintervals, i.e., if (aa,bb) is a
            subinterval of ``(pts[1], pts[2])`` where ``pts[0]`` and ``pts[2]``
            are adjacent elements of ``infodict['pts']``, then (aa,bb) has level l
            if ``|bb-aa| = |pts[2]-pts[1]| * 2**(-l)``.
        'ndin'
            A rank-1 integer array of length P+2. After the first integration
            over the intervals (pts[1], pts[2]), the error estimates over some
            of the intervals may have been increased artificially in order to
            put their subdivision forward. This array has ones in slots
            corresponding to the subintervals for which this happens.
        
        **Weighting the integrand**
        
        The input variables, *weight* and *wvar*, are used to weight the
        integrand by a select list of functions. Different integration
        methods are used to compute the integral with these weighting
        functions, and these do not support specifying break points. The
        possible values of weight and the corresponding weighting functions are.
        
        ==========  ===================================   =====================
        ``weight``  Weight function used                  ``wvar``
        ==========  ===================================   =====================
        'cos'       cos(w*x)                              wvar = w
        'sin'       sin(w*x)                              wvar = w
        'alg'       g(x) = ((x-a)**alpha)*((b-x)**beta)   wvar = (alpha, beta)
        'alg-loga'  g(x)*log(x-a)                         wvar = (alpha, beta)
        'alg-logb'  g(x)*log(b-x)                         wvar = (alpha, beta)
        'alg-log'   g(x)*log(x-a)*log(b-x)                wvar = (alpha, beta)
        'cauchy'    1/(x-c)                               wvar = c
        ==========  ===================================   =====================
        
        wvar holds the parameter w, (alpha, beta), or c depending on the weight
        selected. In these expressions, a and b are the integration limits.
        
        For the 'cos' and 'sin' weighting, additional inputs and outputs are
        available.
        
        For finite integration limits, the integration is performed using a
        Clenshaw-Curtis method which uses Chebyshev moments. For repeated
        calculations, these moments are saved in the output dictionary:
        
        'momcom'
            The maximum level of Chebyshev moments that have been computed,
            i.e., if ``M_c`` is ``infodict['momcom']`` then the moments have been
            computed for intervals of length ``|b-a| * 2**(-l)``,
            ``l=0,1,...,M_c``.
        'nnlog'
            A rank-1 integer array of length M(=limit), containing the
            subdivision levels of the subintervals, i.e., an element of this
            array is equal to l if the corresponding subinterval is
            ``|b-a|* 2**(-l)``.
        'chebmo'
            A rank-2 array of shape (25, maxp1) containing the computed
            Chebyshev moments. These can be passed on to an integration
            over the same interval by passing this array as the second
            element of the sequence wopts and passing infodict['momcom'] as
            the first element.
        
        If one of the integration limits is infinite, then a Fourier integral is
        computed (assuming w neq 0). If full_output is 1 and a numerical error
        is encountered, besides the error message attached to the output tuple,
        a dictionary is also appended to the output tuple which translates the
        error codes in the array ``info['ierlst']`` to English messages. The
        output information dictionary contains the following entries instead of
        'last', 'alist', 'blist', 'rlist', and 'elist':
        
        'lst'
            The number of subintervals needed for the integration (call it ``K_f``).
        'rslst'
            A rank-1 array of length M_f=limlst, whose first ``K_f`` elements
            contain the integral contribution over the interval
            ``(a+(k-1)c, a+kc)`` where ``c = (2*floor(|w|) + 1) * pi / |w|``
            and ``k=1,2,...,K_f``.
        'erlst'
            A rank-1 array of length ``M_f`` containing the error estimate
            corresponding to the interval in the same position in
            ``infodict['rslist']``.
        'ierlst'
            A rank-1 integer array of length ``M_f`` containing an error flag
            corresponding to the interval in the same position in
            ``infodict['rslist']``.  See the explanation dictionary (last entry
            in the output tuple) for the meaning of the codes.
        
        
        **Details of QUADPACK level routines**
        
        `quad` calls routines from the FORTRAN library QUADPACK. This section
        provides details on the conditions for each routine to be called and a
        short description of each routine. The routine called depends on
        `weight`, `points` and the integration limits `a` and `b`.
        
        ================  ==============  ==========  =====================
        QUADPACK routine  `weight`        `points`    infinite bounds
        ================  ==============  ==========  =====================
        qagse             None            No          No
        qagie             None            No          Yes
        qagpe             None            Yes         No
        qawoe             'sin', 'cos'    No          No
        qawfe             'sin', 'cos'    No          either `a` or `b`
        qawse             'alg*'          No          No
        qawce             'cauchy'        No          No
        ================  ==============  ==========  =====================
        
        The following provides a short description from [1]_ for each
        routine.
        
        qagse
            is an integrator based on globally adaptive interval
            subdivision in connection with extrapolation, which will
            eliminate the effects of integrand singularities of
            several types.
        qagie
            handles integration over infinite intervals. The infinite range is
            mapped onto a finite interval and subsequently the same strategy as
            in ``QAGS`` is applied.
        qagpe
            serves the same purposes as QAGS, but also allows the
            user to provide explicit information about the location
            and type of trouble-spots i.e. the abscissae of internal
            singularities, discontinuities and other difficulties of
            the integrand function.
        qawoe
            is an integrator for the evaluation of
            :math:`\int^b_a \cos(\omega x)f(x)dx` or
            :math:`\int^b_a \sin(\omega x)f(x)dx`
            over a finite interval [a,b], where :math:`\omega` and :math:`f`
            are specified by the user. The rule evaluation component is based
            on the modified Clenshaw-Curtis technique
        
            An adaptive subdivision scheme is used in connection
            with an extrapolation procedure, which is a modification
            of that in ``QAGS`` and allows the algorithm to deal with
            singularities in :math:`f(x)`.
        qawfe
            calculates the Fourier transform
            :math:`\int^\infty_a \cos(\omega x)f(x)dx` or
            :math:`\int^\infty_a \sin(\omega x)f(x)dx`
            for user-provided :math:`\omega` and :math:`f`. The procedure of
            ``QAWO`` is applied on successive finite intervals, and convergence
            acceleration by means of the :math:`\varepsilon`-algorithm is applied
            to the series of integral approximations.
        qawse
            approximate :math:`\int^b_a w(x)f(x)dx`, with :math:`a < b` where
            :math:`w(x) = (x-a)^{\alpha}(b-x)^{\beta}v(x)` with
            :math:`\alpha,\beta > -1`, where :math:`v(x)` may be one of the
            following functions: :math:`1`, :math:`\log(x-a)`, :math:`\log(b-x)`,
            :math:`\log(x-a)\log(b-x)`.
        
            The user specifies :math:`\alpha`, :math:`\beta` and the type of the
            function :math:`v`. A globally adaptive subdivision strategy is
            applied, with modified Clenshaw-Curtis integration on those
            subintervals which contain `a` or `b`.
        qawce
            compute :math:`\int^b_a f(x) / (x-c)dx` where the integral must be
            interpreted as a Cauchy principal value integral, for user specified
            :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
            Clenshaw-Curtis integration is used on those intervals containing the
            point :math:`x = c`.
        
        **Integration of Complex Function of a Real Variable**
        
        A complex valued function, :math:`f`, of a real variable can be written as
        :math:`f = g + ih`.  Similarly, the integral of :math:`f` can be
        written as
        
        .. math::
            \int_a^b f(x) dx = \int_a^b g(x) dx + i\int_a^b h(x) dx
        
        assuming that the integrals of :math:`g` and :math:`h` exist
        over the interval :math:`[a,b]` [2]_. Therefore, ``quad`` integrates
        complex-valued functions by integrating the real and imaginary components
        separately.
        
        """
    def quad_vec(self, a, b, epsabs=1e-200, epsrel=1e-08, norm='2', cache_size=100000000.0, limit=10000, workers=1, points=None, quadrature=None, full_output=False, args=()):
        r"""Adaptive integration of a vector-valued function.
        
        Parameters
        ----------
        f : callable
            Vector-valued function f(x) to integrate.
        a : float
            Initial point.
        b : float
            Final point.
        epsabs : float, optional
            Absolute tolerance.
        epsrel : float, optional
            Relative tolerance.
        norm : {'max', '2'}, optional
            Vector norm to use for error estimation.
        cache_size : int, optional
            Number of bytes to use for memoization.
        limit : float or int, optional
            An upper bound on the number of subintervals used in the adaptive
            algorithm.
        workers : int or map-like callable, optional
            If `workers` is an integer, part of the computation is done in
            parallel subdivided to this many tasks (using
            :class:`python:multiprocessing.pool.Pool`).
            Supply `-1` to use all cores available to the Process.
            Alternatively, supply a map-like callable, such as
            :meth:`python:multiprocessing.pool.Pool.map` for evaluating the
            population in parallel.
            This evaluation is carried out as ``workers(func, iterable)``.
        points : list, optional
            List of additional breakpoints.
        quadrature : {'gk21', 'gk15', 'trapezoid'}, optional
            Quadrature rule to use on subintervals.
            Options: 'gk21' (Gauss-Kronrod 21-point rule),
            'gk15' (Gauss-Kronrod 15-point rule),
            'trapezoid' (composite trapezoid rule).
            Default: 'gk21' for finite intervals and 'gk15' for (semi-)infinite
        full_output : bool, optional
            Return an additional ``info`` dictionary.
        args : tuple, optional
            Extra arguments to pass to function, if any.
        
            .. versionadded:: 1.8.0
        
        Returns
        -------
        res : {float, array-like}
            Estimate for the result
        err : float
            Error estimate for the result in the given norm
        info : dict
            Returned only when ``full_output=True``.
            Info dictionary. Is an object with the attributes:
        
                success : bool
                    Whether integration reached target precision.
                status : int
                    Indicator for convergence, success (0),
                    failure (1), and failure due to rounding error (2).
                neval : int
                    Number of function evaluations.
                intervals : ndarray, shape (num_intervals, 2)
                    Start and end points of subdivision intervals.
                integrals : ndarray, shape (num_intervals, ...)
                    Integral for each interval.
                    Note that at most ``cache_size`` values are recorded,
                    and the array may contains *nan* for missing items.
                errors : ndarray, shape (num_intervals,)
                    Estimated integration error for each interval.
        
        Notes
        -----
        The algorithm mainly follows the implementation of QUADPACK's
        DQAG* algorithms, implementing global error control and adaptive
        subdivision.
        
        The algorithm here has some differences to the QUADPACK approach:
        
        Instead of subdividing one interval at a time, the algorithm
        subdivides N intervals with largest errors at once. This enables
        (partial) parallelization of the integration.
        
        The logic of subdividing "next largest" intervals first is then
        not implemented, and we rely on the above extension to avoid
        concentrating on "small" intervals only.
        
        The Wynn epsilon table extrapolation is not used (QUADPACK uses it
        for infinite intervals). This is because the algorithm here is
        supposed to work on vector-valued functions, in an user-specified
        norm, and the extension of the epsilon algorithm to this case does
        not appear to be widely agreed. For max-norm, using elementwise
        Wynn epsilon could be possible, but we do not do this here with
        the hope that the epsilon extrapolation is mainly useful in
        special cases.
        """
    def ridder(self, a, b, args=(), xtol=2e-12, rtol=np.float64(8.881784197001252e-16), maxiter=100, full_output=False, disp=True):
        r"""
        Find a root of a function in an interval using Ridder's method.
        
        Parameters
        ----------
        f : function
            Python function returning a number. f must be continuous, and f(a) and
            f(b) must have opposite signs.
        a : scalar
            One end of the bracketing interval [a,b].
        b : scalar
            The other end of the bracketing interval [a,b].
        xtol : number, optional
            The computed root ``x0`` will satisfy ``np.allclose(x, x0,
            atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
            parameter must be positive.
        rtol : number, optional
            The computed root ``x0`` will satisfy ``np.allclose(x, x0,
            atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
            parameter cannot be smaller than its default value of
            ``4*np.finfo(float).eps``.
        maxiter : int, optional
            If convergence is not achieved in `maxiter` iterations, an error is
            raised. Must be >= 0.
        args : tuple, optional
            Containing extra arguments for the function `f`.
            `f` is called by ``apply(f, (x)+args)``.
        full_output : bool, optional
            If `full_output` is False, the root is returned. If `full_output` is
            True, the return value is ``(x, r)``, where `x` is the root, and `r` is
            a `RootResults` object.
        disp : bool, optional
            If True, raise RuntimeError if the algorithm didn't converge.
            Otherwise, the convergence status is recorded in any `RootResults`
            return object.
        
        Returns
        -------
        root : float
            Root of `f` between `a` and `b`.
        r : `RootResults` (present if ``full_output = True``)
            Object containing information about the convergence.
            In particular, ``r.converged`` is True if the routine converged.
        
        See Also
        --------
        brentq, brenth, bisect, newton : 1-D root-finding
        fixed_point : scalar fixed-point finder
        
        Notes
        -----
        Uses [Ridders1979]_ method to find a root of the function `f` between the
        arguments `a` and `b`. Ridders' method is faster than bisection, but not
        generally as fast as the Brent routines. [Ridders1979]_ provides the
        classic description and source of the algorithm. A description can also be
        found in any recent edition of Numerical Recipes.
        
        The routine used here diverges slightly from standard presentations in
        order to be a bit more careful of tolerance.
        """
    def root(self, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None):
        r"""
        Find a root of a vector function.
        
        Parameters
        ----------
        fun : callable
            A vector function to find a root of.
        
            Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
            ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
            Rather than passing ``f0`` as the callable, wrap it to accept
            only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
            callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
            gathered before invoking this function.
        x0 : ndarray
            Initial guess.
        args : tuple, optional
            Extra arguments passed to the objective function and its Jacobian.
        method : str, optional
            Type of solver. Should be one of
        
            - 'hybr'             :ref:`(see here) <optimize.root-hybr>`
            - 'lm'               :ref:`(see here) <optimize.root-lm>`
            - 'broyden1'         :ref:`(see here) <optimize.root-broyden1>`
            - 'broyden2'         :ref:`(see here) <optimize.root-broyden2>`
            - 'anderson'         :ref:`(see here) <optimize.root-anderson>`
            - 'linearmixing'     :ref:`(see here) <optimize.root-linearmixing>`
            - 'diagbroyden'      :ref:`(see here) <optimize.root-diagbroyden>`
            - 'excitingmixing'   :ref:`(see here) <optimize.root-excitingmixing>`
            - 'krylov'           :ref:`(see here) <optimize.root-krylov>`
            - 'df-sane'          :ref:`(see here) <optimize.root-dfsane>`
        
        jac : bool or callable, optional
            If `jac` is a Boolean and is True, `fun` is assumed to return the
            value of Jacobian along with the objective function. If False, the
            Jacobian will be estimated numerically.
            `jac` can also be a callable returning the Jacobian of `fun`. In
            this case, it must accept the same arguments as `fun`.
        tol : float, optional
            Tolerance for termination. For detailed control, use solver-specific
            options.
        callback : function, optional
            Optional callback function. It is called on every iteration as
            ``callback(x, f)`` where `x` is the current solution and `f`
            the corresponding residual. For all methods but 'hybr' and 'lm'.
        options : dict, optional
            A dictionary of solver options. E.g., `xtol` or `maxiter`, see
            :obj:`show_options()` for details.
        
        Returns
        -------
        sol : OptimizeResult
            The solution represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the algorithm exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.
        
        See also
        --------
        show_options : Additional options accepted by the solvers
        
        Notes
        -----
        This section describes the available solvers that can be selected by the
        'method' parameter. The default method is *hybr*.
        
        Method *hybr* uses a modification of the Powell hybrid method as
        implemented in MINPACK [1]_.
        
        Method *lm* solves the system of nonlinear equations in a least squares
        sense using a modification of the Levenberg-Marquardt algorithm as
        implemented in MINPACK [1]_.
        
        Method *df-sane* is a derivative-free spectral method. [3]_
        
        Methods *broyden1*, *broyden2*, *anderson*, *linearmixing*,
        *diagbroyden*, *excitingmixing*, *krylov* are inexact Newton methods,
        with backtracking or full line searches [2]_. Each method corresponds
        to a particular Jacobian approximations.
        
        - Method *broyden1* uses Broyden's first Jacobian approximation, it is
          known as Broyden's good method.
        - Method *broyden2* uses Broyden's second Jacobian approximation, it
          is known as Broyden's bad method.
        - Method *anderson* uses (extended) Anderson mixing.
        - Method *Krylov* uses Krylov approximation for inverse Jacobian. It
          is suitable for large-scale problem.
        - Method *diagbroyden* uses diagonal Broyden Jacobian approximation.
        - Method *linearmixing* uses a scalar Jacobian approximation.
        - Method *excitingmixing* uses a tuned diagonal Jacobian
          approximation.
        
        .. warning::
        
            The algorithms implemented for methods *diagbroyden*,
            *linearmixing* and *excitingmixing* may be useful for specific
            problems, but whether they will work may depend strongly on the
            problem.
        
        .. versionadded:: 0.11.0
        """
    def root_scalar(self, args=(), method=None, bracket=None, fprime=None, fprime2=None, x0=None, x1=None, xtol=None, rtol=None, maxiter=None, options=None):
        r"""
        Find a root of a scalar function.
        
        Parameters
        ----------
        f : callable
            A function to find a root of.
        
            Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
            ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
            Rather than passing ``f0`` as the callable, wrap it to accept
            only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
            callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
            gathered before invoking this function.
        args : tuple, optional
            Extra arguments passed to the objective function and its derivative(s).
        method : str, optional
            Type of solver.  Should be one of
        
            - 'bisect'    :ref:`(see here) <optimize.root_scalar-bisect>`
            - 'brentq'    :ref:`(see here) <optimize.root_scalar-brentq>`
            - 'brenth'    :ref:`(see here) <optimize.root_scalar-brenth>`
            - 'ridder'    :ref:`(see here) <optimize.root_scalar-ridder>`
            - 'toms748'    :ref:`(see here) <optimize.root_scalar-toms748>`
            - 'newton'    :ref:`(see here) <optimize.root_scalar-newton>`
            - 'secant'    :ref:`(see here) <optimize.root_scalar-secant>`
            - 'halley'    :ref:`(see here) <optimize.root_scalar-halley>`
        
        bracket: A sequence of 2 floats, optional
            An interval bracketing a root.  ``f(x, *args)`` must have different
            signs at the two endpoints.
        x0 : float, optional
            Initial guess.
        x1 : float, optional
            A second guess.
        fprime : bool or callable, optional
            If `fprime` is a boolean and is True, `f` is assumed to return the
            value of the objective function and of the derivative.
            `fprime` can also be a callable returning the derivative of `f`. In
            this case, it must accept the same arguments as `f`.
        fprime2 : bool or callable, optional
            If `fprime2` is a boolean and is True, `f` is assumed to return the
            value of the objective function and of the
            first and second derivatives.
            `fprime2` can also be a callable returning the second derivative of `f`.
            In this case, it must accept the same arguments as `f`.
        xtol : float, optional
            Tolerance (absolute) for termination.
        rtol : float, optional
            Tolerance (relative) for termination.
        maxiter : int, optional
            Maximum number of iterations.
        options : dict, optional
            A dictionary of solver options. E.g., ``k``, see
            :obj:`show_options()` for details.
        
        Returns
        -------
        sol : RootResults
            The solution represented as a ``RootResults`` object.
            Important attributes are: ``root`` the solution , ``converged`` a
            boolean flag indicating if the algorithm exited successfully and
            ``flag`` which describes the cause of the termination. See
            `RootResults` for a description of other attributes.
        
        See also
        --------
        show_options : Additional options accepted by the solvers
        root : Find a root of a vector function.
        
        Notes
        -----
        This section describes the available solvers that can be selected by the
        'method' parameter.
        
        The default is to use the best method available for the situation
        presented.
        If a bracket is provided, it may use one of the bracketing methods.
        If a derivative and an initial value are specified, it may
        select one of the derivative-based methods.
        If no method is judged applicable, it will raise an Exception.
        
        Arguments for each method are as follows (x=required, o=optional).
        
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        |                    method                     | f | args | bracket | x0 | x1 | fprime | fprime2 | xtol | rtol | maxiter | options |
        +===============================================+===+======+=========+====+====+========+=========+======+======+=========+=========+
        | :ref:`bisect <optimize.root_scalar-bisect>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`brentq <optimize.root_scalar-brentq>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`brenth <optimize.root_scalar-brenth>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`ridder <optimize.root_scalar-ridder>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`toms748 <optimize.root_scalar-toms748>` | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`secant <optimize.root_scalar-secant>`   | x |  o   |         | x  | o  |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`newton <optimize.root_scalar-newton>`   | x |  o   |         | x  |    |   o    |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`halley <optimize.root_scalar-halley>`   | x |  o   |         | x  |    |   x    |    x    |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        """
    def shgo(self, bounds, args=(), constraints=None, n=100, iters=1, callback=None, minimizer_kwargs=None, options=None, sampling_method='simplicial', workers=1):
        r"""
        Finds the global minimum of a function using SHG optimization.
        
        SHGO stands for "simplicial homology global optimization".
        
        Parameters
        ----------
        func : callable
            The objective function to be minimized.  Must be in the form
            ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
            and ``args`` is a tuple of any additional fixed parameters needed to
            completely specify the function.
        bounds : sequence or `Bounds`
            Bounds for variables. There are two ways to specify the bounds:
        
            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`.
        
        args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            objective function.
        constraints : {Constraint, dict} or List of {Constraint, dict}, optional
            Constraints definition. Only for COBYLA, COBYQA, SLSQP and trust-constr.
            See the tutorial [5]_ for further details on specifying constraints.
        
            .. note::
        
               Only COBYLA, COBYQA, SLSQP, and trust-constr local minimize methods
               currently support constraint arguments. If the ``constraints``
               sequence used in the local optimization problem is not defined in
               ``minimizer_kwargs`` and a constrained method is used then the
               global ``constraints`` will be used.
               (Defining a ``constraints`` sequence in ``minimizer_kwargs``
               means that ``constraints`` will not be added so if equality
               constraints and so forth need to be added then the inequality
               functions in ``constraints`` need to be added to
               ``minimizer_kwargs`` too).
               COBYLA only supports inequality constraints.
        
            .. versionchanged:: 1.11.0
        
               ``constraints`` accepts `NonlinearConstraint`, `LinearConstraint`.
        
        n : int, optional
            Number of sampling points used in the construction of the simplicial
            complex. For the default ``simplicial`` sampling method 2**dim + 1
            sampling points are generated instead of the default ``n=100``. For all
            other specified values `n` sampling points are generated. For
            ``sobol``, ``halton`` and other arbitrary `sampling_methods` ``n=100`` or
            another specified number of sampling points are generated.
        iters : int, optional
            Number of iterations used in the construction of the simplicial
            complex. Default is 1.
        callback : callable, optional
            Called after each iteration, as ``callback(xk)``, where ``xk`` is the
            current parameter vector.
        minimizer_kwargs : dict, optional
            Extra keyword arguments to be passed to the minimizer
            ``scipy.optimize.minimize``. Some important options could be:
        
            method : str
                The minimization method. If not given, chosen to be one of
                BFGS, L-BFGS-B, SLSQP, depending on whether or not the
                problem has constraints or bounds.
            args : tuple
                Extra arguments passed to the objective function (``func``) and
                its derivatives (Jacobian, Hessian).
            options : dict, optional
                Note that by default the tolerance is specified as
                ``{ftol: 1e-12}``
        
        options : dict, optional
            A dictionary of solver options. Many of the options specified for the
            global routine are also passed to the ``scipy.optimize.minimize``
            routine. The options that are also passed to the local routine are
            marked with "(L)".
        
            Stopping criteria, the algorithm will terminate if any of the specified
            criteria are met. However, the default algorithm does not require any
            to be specified:
        
            maxfev : int (L)
                Maximum number of function evaluations in the feasible domain.
                (Note only methods that support this option will terminate
                the routine at precisely exact specified value. Otherwise the
                criterion will only terminate during a global iteration)
            f_min : float
                Specify the minimum objective function value, if it is known.
            f_tol : float
                Precision goal for the value of f in the stopping
                criterion. Note that the global routine will also
                terminate if a sampling point in the global routine is
                within this tolerance.
            maxiter : int
                Maximum number of iterations to perform.
            maxev : int
                Maximum number of sampling evaluations to perform (includes
                searching in infeasible points).
            maxtime : float
                Maximum processing runtime allowed
            minhgrd : int
                Minimum homology group rank differential. The homology group of the
                objective function is calculated (approximately) during every
                iteration. The rank of this group has a one-to-one correspondence
                with the number of locally convex subdomains in the objective
                function (after adequate sampling points each of these subdomains
                contain a unique global minimum). If the difference in the hgr is 0
                between iterations for ``maxhgrd`` specified iterations the
                algorithm will terminate.
        
            Objective function knowledge:
        
            symmetry : list or bool
                Specify if the objective function contains symmetric variables.
                The search space (and therefore performance) is decreased by up to
                O(n!) times in the fully symmetric case. If `True` is specified
                then all variables will be set symmetric to the first variable.
                Default
                is set to False.
        
                E.g.  f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2
        
                In this equation x_2 and x_3 are symmetric to x_1, while x_5 and
                x_6 are symmetric to x_4, this can be specified to the solver as::
        
                    symmetry = [0,  # Variable 1
                                0,  # symmetric to variable 1
                                0,  # symmetric to variable 1
                                3,  # Variable 4
                                3,  # symmetric to variable 4
                                3,  # symmetric to variable 4
                                ]
        
            jac : bool or callable, optional
                Jacobian (gradient) of objective function. Only for CG, BFGS,
                Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg. If ``jac`` is a
                boolean and is True, ``fun`` is assumed to return the gradient
                along with the objective function. If False, the gradient will be
                estimated numerically. ``jac`` can also be a callable returning the
                gradient of the objective. In this case, it must accept the same
                arguments as ``fun``. (Passed to `scipy.optimize.minimize`
                automatically)
        
            hess, hessp : callable, optional
                Hessian (matrix of second-order derivatives) of objective function
                or Hessian of objective function times an arbitrary vector p.
                Only for Newton-CG, dogleg, trust-ncg. Only one of ``hessp`` or
                ``hess`` needs to be given. If ``hess`` is provided, then
                ``hessp`` will be ignored. If neither ``hess`` nor ``hessp`` is
                provided, then the Hessian product will be approximated using
                finite differences on ``jac``. ``hessp`` must compute the Hessian
                times an arbitrary vector. (Passed to `scipy.optimize.minimize`
                automatically)
        
            Algorithm settings:
        
            minimize_every_iter : bool
                If True then promising global sampling points will be passed to a
                local minimization routine every iteration. If True then only the
                final minimizer pool will be run. Defaults to True.
        
            local_iter : int
                Only evaluate a few of the best minimizer pool candidates every
                iteration. If False all potential points are passed to the local
                minimization routine.
        
            infty_constraints : bool
                If True then any sampling points generated which are outside will
                the feasible domain will be saved and given an objective function
                value of ``inf``. If False then these points will be discarded.
                Using this functionality could lead to higher performance with
                respect to function evaluations before the global minimum is found,
                specifying False will use less memory at the cost of a slight
                decrease in performance. Defaults to True.
        
            Feedback:
        
            disp : bool (L)
                Set to True to print convergence messages.
        
        sampling_method : str or function, optional
            Current built in sampling method options are ``halton``, ``sobol`` and
            ``simplicial``. The default ``simplicial`` provides
            the theoretical guarantee of convergence to the global minimum in
            finite time. ``halton`` and ``sobol`` method are faster in terms of
            sampling point generation at the cost of the loss of
            guaranteed convergence. It is more appropriate for most "easier"
            problems where the convergence is relatively fast.
            User defined sampling functions must accept two arguments of ``n``
            sampling points of dimension ``dim`` per call and output an array of
            sampling points with shape `n x dim`.
        
        workers : int or map-like callable, optional
            Sample and run the local serial minimizations in parallel.
            Supply -1 to use all available CPU cores, or an int to use
            that many Processes (uses `multiprocessing.Pool <multiprocessing>`).
        
            Alternatively supply a map-like callable, such as
            `multiprocessing.Pool.map` for parallel evaluation.
            This evaluation is carried out as ``workers(func, iterable)``.
            Requires that `func` be pickleable.
        
            .. versionadded:: 1.11.0
        
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            Important attributes are:
            ``x`` the solution array corresponding to the global minimum,
            ``fun`` the function output at the global solution,
            ``xl`` an ordered list of local minima solutions,
            ``funl`` the function output at the corresponding local solutions,
            ``success`` a Boolean flag indicating if the optimizer exited
            successfully,
            ``message`` which describes the cause of the termination,
            ``nfev`` the total number of objective function evaluations including
            the sampling calls,
            ``nlfev`` the total number of objective function evaluations
            culminating from all local search optimizations,
            ``nit`` number of iterations performed by the global routine.
        
        Notes
        -----
        Global optimization using simplicial homology global optimization [1]_.
        Appropriate for solving general purpose NLP and blackbox optimization
        problems to global optimality (low-dimensional problems).
        
        In general, the optimization problems are of the form::
        
            minimize f(x) subject to
        
            g_i(x) >= 0,  i = 1,...,m
            h_j(x)  = 0,  j = 1,...,p
        
        where x is a vector of one or more variables. ``f(x)`` is the objective
        function ``R^n -> R``, ``g_i(x)`` are the inequality constraints, and
        ``h_j(x)`` are the equality constraints.
        
        Optionally, the lower and upper bounds for each element in x can also be
        specified using the `bounds` argument.
        
        While most of the theoretical advantages of SHGO are only proven for when
        ``f(x)`` is a Lipschitz smooth function, the algorithm is also proven to
        converge to the global optimum for the more general case where ``f(x)`` is
        non-continuous, non-convex and non-smooth, if the default sampling method
        is used [1]_.
        
        The local search method may be specified using the ``minimizer_kwargs``
        parameter which is passed on to ``scipy.optimize.minimize``. By default,
        the ``SLSQP`` method is used. In general, it is recommended to use the
        ``SLSQP``, ``COBYLA``, or ``COBYQA`` local minimization if inequality
        constraints are defined for the problem since the other methods do not use
        constraints.
        
        The ``halton`` and ``sobol`` method points are generated using
        `scipy.stats.qmc`. Any other QMC method could be used.
        """
    def solve(self, args=(), method=None, bracket=None, fprime=None, fprime2=None, x0=None, x1=None, xtol=None, rtol=None, maxiter=None, options=None):
        r"""
        Find a root of a scalar function.
        
        Parameters
        ----------
        f : callable
            A function to find a root of.
        
            Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
            ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
            Rather than passing ``f0`` as the callable, wrap it to accept
            only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
            callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
            gathered before invoking this function.
        args : tuple, optional
            Extra arguments passed to the objective function and its derivative(s).
        method : str, optional
            Type of solver.  Should be one of
        
            - 'bisect'    :ref:`(see here) <optimize.root_scalar-bisect>`
            - 'brentq'    :ref:`(see here) <optimize.root_scalar-brentq>`
            - 'brenth'    :ref:`(see here) <optimize.root_scalar-brenth>`
            - 'ridder'    :ref:`(see here) <optimize.root_scalar-ridder>`
            - 'toms748'    :ref:`(see here) <optimize.root_scalar-toms748>`
            - 'newton'    :ref:`(see here) <optimize.root_scalar-newton>`
            - 'secant'    :ref:`(see here) <optimize.root_scalar-secant>`
            - 'halley'    :ref:`(see here) <optimize.root_scalar-halley>`
        
        bracket: A sequence of 2 floats, optional
            An interval bracketing a root.  ``f(x, *args)`` must have different
            signs at the two endpoints.
        x0 : float, optional
            Initial guess.
        x1 : float, optional
            A second guess.
        fprime : bool or callable, optional
            If `fprime` is a boolean and is True, `f` is assumed to return the
            value of the objective function and of the derivative.
            `fprime` can also be a callable returning the derivative of `f`. In
            this case, it must accept the same arguments as `f`.
        fprime2 : bool or callable, optional
            If `fprime2` is a boolean and is True, `f` is assumed to return the
            value of the objective function and of the
            first and second derivatives.
            `fprime2` can also be a callable returning the second derivative of `f`.
            In this case, it must accept the same arguments as `f`.
        xtol : float, optional
            Tolerance (absolute) for termination.
        rtol : float, optional
            Tolerance (relative) for termination.
        maxiter : int, optional
            Maximum number of iterations.
        options : dict, optional
            A dictionary of solver options. E.g., ``k``, see
            :obj:`show_options()` for details.
        
        Returns
        -------
        sol : RootResults
            The solution represented as a ``RootResults`` object.
            Important attributes are: ``root`` the solution , ``converged`` a
            boolean flag indicating if the algorithm exited successfully and
            ``flag`` which describes the cause of the termination. See
            `RootResults` for a description of other attributes.
        
        See also
        --------
        show_options : Additional options accepted by the solvers
        root : Find a root of a vector function.
        
        Notes
        -----
        This section describes the available solvers that can be selected by the
        'method' parameter.
        
        The default is to use the best method available for the situation
        presented.
        If a bracket is provided, it may use one of the bracketing methods.
        If a derivative and an initial value are specified, it may
        select one of the derivative-based methods.
        If no method is judged applicable, it will raise an Exception.
        
        Arguments for each method are as follows (x=required, o=optional).
        
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        |                    method                     | f | args | bracket | x0 | x1 | fprime | fprime2 | xtol | rtol | maxiter | options |
        +===============================================+===+======+=========+====+====+========+=========+======+======+=========+=========+
        | :ref:`bisect <optimize.root_scalar-bisect>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`brentq <optimize.root_scalar-brentq>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`brenth <optimize.root_scalar-brenth>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`ridder <optimize.root_scalar-ridder>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`toms748 <optimize.root_scalar-toms748>` | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`secant <optimize.root_scalar-secant>`   | x |  o   |         | x  | o  |        |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`newton <optimize.root_scalar-newton>`   | x |  o   |         | x  |    |   o    |         |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        | :ref:`halley <optimize.root_scalar-halley>`   | x |  o   |         | x  |    |   x    |    x    |  o   |  o   |    o    |   o     |
        +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
        """
    def tanhsinh(self, a, b, args=(), log=False, maxlevel=None, minlevel=2, atol=None, rtol=None, preserve_shape=False, callback=None):
        r"""Evaluate a convergent integral numerically using tanh-sinh quadrature.
        
        In practice, tanh-sinh quadrature achieves quadratic convergence for
        many integrands: the number of accurate *digits* scales roughly linearly
        with the number of function evaluations [1]_.
        
        Either or both of the limits of integration may be infinite, and
        singularities at the endpoints are acceptable. Divergent integrals and
        integrands with non-finite derivatives or singularities within an interval
        are out of scope, but the latter may be evaluated be calling `tanhsinh` on
        each sub-interval separately.
        
        Parameters
        ----------
        f : callable
            The function to be integrated. The signature must be::
        
                f(xi: ndarray, *argsi) -> ndarray
        
            where each element of ``xi`` is a finite real number and ``argsi`` is a tuple,
            which may contain an arbitrary number of arrays that are broadcastable
            with ``xi``. `f` must be an elementwise function: see documentation of parameter
            `preserve_shape` for details. It must not mutate the array ``xi`` or the arrays
            in ``argsi``.
            If ``f`` returns a value with complex dtype when evaluated at
            either endpoint, subsequent arguments ``x`` will have complex dtype
            (but zero imaginary part).
        a, b : float array_like
            Real lower and upper limits of integration. Must be broadcastable with one
            another and with arrays in `args`. Elements may be infinite.
        args : tuple of array_like, optional
            Additional positional array arguments to be passed to `f`. Arrays
            must be broadcastable with one another and the arrays of `a` and `b`.
            If the callable for which the root is desired requires arguments that are
            not broadcastable with `x`, wrap that callable with `f` such that `f`
            accepts only `x` and broadcastable ``*args``.
        log : bool, default: False
            Setting to True indicates that `f` returns the log of the integrand
            and that `atol` and `rtol` are expressed as the logs of the absolute
            and relative errors. In this case, the result object will contain the
            log of the integral and error. This is useful for integrands for which
            numerical underflow or overflow would lead to inaccuracies.
            When ``log=True``, the integrand (the exponential of `f`) must be real,
            but it may be negative, in which case the log of the integrand is a
            complex number with an imaginary part that is an odd multiple of .
        maxlevel : int, default: 10
            The maximum refinement level of the algorithm.
        
            At the zeroth level, `f` is called once, performing 16 function
            evaluations. At each subsequent level, `f` is called once more,
            approximately doubling the number of function evaluations that have
            been performed. Accordingly, for many integrands, each successive level
            will double the number of accurate digits in the result (up to the
            limits of floating point precision).
        
            The algorithm will terminate after completing level `maxlevel` or after
            another termination condition is satisfied, whichever comes first.
        minlevel : int, default: 2
            The level at which to begin iteration (default: 2). This does not
            change the total number of function evaluations or the abscissae at
            which the function is evaluated; it changes only the *number of times*
            `f` is called. If ``minlevel=k``, then the integrand is evaluated at
            all abscissae from levels ``0`` through ``k`` in a single call.
            Note that if `minlevel` exceeds `maxlevel`, the provided `minlevel` is
            ignored, and `minlevel` is set equal to `maxlevel`.
        atol, rtol : float, optional
            Absolute termination tolerance (default: 0) and relative termination
            tolerance (default: ``eps**0.75``, where ``eps`` is the precision of
            the result dtype), respectively.  Iteration will stop when
            ``res.error < atol`` or  ``res.error < res.integral * rtol``. The error
            estimate is as described in [1]_ Section 5 but with a lower bound of
            ``eps * res.integral``. While not theoretically rigorous or
            conservative, it is said to work well in practice. Must be non-negative
            and finite if `log` is False, and must be expressed as the log of a
            non-negative and finite number if `log` is True.
        preserve_shape : bool, default: False
            In the following, "arguments of `f`" refers to the array ``xi`` and
            any arrays within ``argsi``. Let ``shape`` be the broadcasted shape
            of `a`, `b`, and all elements of `args` (which is conceptually
            distinct from ``xi` and ``argsi`` passed into `f`).
        
            - When ``preserve_shape=False`` (default), `f` must accept arguments
              of *any* broadcastable shapes.
        
            - When ``preserve_shape=True``, `f` must accept arguments of shape
              ``shape`` *or* ``shape + (n,)``, where ``(n,)`` is the number of
              abscissae at which the function is being evaluated.
        
            In either case, for each scalar element ``xi[j]`` within ``xi``, the array
            returned by `f` must include the scalar ``f(xi[j])`` at the same index.
            Consequently, the shape of the output is always the shape of the input
            ``xi``.
        
            See Examples.
        
        callback : callable, optional
            An optional user-supplied function to be called before the first
            iteration and after each iteration.
            Called as ``callback(res)``, where ``res`` is a ``_RichResult``
            similar to that returned by `_differentiate` (but containing the
            current iterate's values of all variables). If `callback` raises a
            ``StopIteration``, the algorithm will terminate immediately and
            `tanhsinh` will return a result object. `callback` must not mutate
            `res` or its attributes.
        
        Returns
        -------
        res : _RichResult
            An object similar to an instance of `scipy.optimize.OptimizeResult` with the
            following attributes. (The descriptions are written as though the values will
            be scalars; however, if `f` returns an array, the outputs will be
            arrays of the same shape.)
        
            success : bool array
                ``True`` when the algorithm terminated successfully (status ``0``).
                ``False`` otherwise.
            status : int array
                An integer representing the exit status of the algorithm.
        
                ``0`` : The algorithm converged to the specified tolerances.
                ``-1`` : (unused)
                ``-2`` : The maximum number of iterations was reached.
                ``-3`` : A non-finite value was encountered.
                ``-4`` : Iteration was terminated by `callback`.
                ``1`` : The algorithm is proceeding normally (in `callback` only).
        
            integral : float array
                An estimate of the integral.
            error : float array
                An estimate of the error. Only available if level two or higher
                has been completed; otherwise NaN.
            maxlevel : int array
                The maximum refinement level used.
            nfev : int array
                The number of points at which `f` was evaluated.
        
        See Also
        --------
        quad
        
        Notes
        -----
        Implements the algorithm as described in [1]_ with minor adaptations for
        finite-precision arithmetic, including some described by [2]_ and [3]_. The
        tanh-sinh scheme was originally introduced in [4]_.
        
        Due to floating-point error in the abscissae, the function may be evaluated
        at the endpoints of the interval during iterations, but the values returned by
        the function at the endpoints will be ignored.
        """
    def toms748(self, a, b, args=(), k=1, xtol=2e-12, rtol=np.float64(8.881784197001252e-16), maxiter=100, full_output=False, disp=True):
        r"""
        Find a root using TOMS Algorithm 748 method.
        
        Implements the Algorithm 748 method of Alefeld, Potro and Shi to find a
        root of the function `f` on the interval ``[a , b]``, where ``f(a)`` and
        `f(b)` must have opposite signs.
        
        It uses a mixture of inverse cubic interpolation and
        "Newton-quadratic" steps. [APS1995].
        
        Parameters
        ----------
        f : function
            Python function returning a scalar. The function :math:`f`
            must be continuous, and :math:`f(a)` and :math:`f(b)`
            have opposite signs.
        a : scalar,
            lower boundary of the search interval
        b : scalar,
            upper boundary of the search interval
        args : tuple, optional
            containing extra arguments for the function `f`.
            `f` is called by ``f(x, *args)``.
        k : int, optional
            The number of Newton quadratic steps to perform each
            iteration. ``k>=1``.
        xtol : scalar, optional
            The computed root ``x0`` will satisfy ``np.allclose(x, x0,
            atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
            parameter must be positive.
        rtol : scalar, optional
            The computed root ``x0`` will satisfy ``np.allclose(x, x0,
            atol=xtol, rtol=rtol)``, where ``x`` is the exact root.
        maxiter : int, optional
            If convergence is not achieved in `maxiter` iterations, an error is
            raised. Must be >= 0.
        full_output : bool, optional
            If `full_output` is False, the root is returned. If `full_output` is
            True, the return value is ``(x, r)``, where `x` is the root, and `r` is
            a `RootResults` object.
        disp : bool, optional
            If True, raise RuntimeError if the algorithm didn't converge.
            Otherwise, the convergence status is recorded in the `RootResults`
            return object.
        
        Returns
        -------
        root : float
            Approximate root of `f`
        r : `RootResults` (present if ``full_output = True``)
            Object containing information about the convergence. In particular,
            ``r.converged`` is True if the routine converged.
        
        See Also
        --------
        brentq, brenth, ridder, bisect, newton
        fsolve : find roots in N dimensions.
        
        Notes
        -----
        `f` must be continuous.
        Algorithm 748 with ``k=2`` is asymptotically the most efficient
        algorithm known for finding roots of a four times continuously
        differentiable function.
        In contrast with Brent's algorithm, which may only decrease the length of
        the enclosing bracket on the last step, Algorithm 748 decreases it each
        iteration with the same asymptotic efficiency as it finds the root.
        
        For easy statement of efficiency indices, assume that `f` has 4
        continuous deriviatives.
        For ``k=1``, the convergence order is at least 2.7, and with about
        asymptotically 2 function evaluations per iteration, the efficiency
        index is approximately 1.65.
        For ``k=2``, the order is about 4.6 with asymptotically 3 function
        evaluations per iteration, and the efficiency index 1.66.
        For higher values of `k`, the efficiency index approaches
        the kth root of ``(3k-2)``, hence ``k=1`` or ``k=2`` are
        usually appropriate.
        """
    def tplquad(self, a, b, gfun, hfun, qfun, rfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08):
        r"""
        Compute a triple (definite) integral.
        
        Return the triple integral of ``func(z, y, x)`` from ``x = a..b``,
        ``y = gfun(x)..hfun(x)``, and ``z = qfun(x,y)..rfun(x,y)``.
        
        Parameters
        ----------
        func : function
            A Python function or method of at least three variables in the
            order (z, y, x).
        a, b : float
            The limits of integration in x: `a` < `b`
        gfun : function or float
            The lower boundary curve in y which is a function taking a single
            floating point argument (x) and returning a floating point result
            or a float indicating a constant boundary curve.
        hfun : function or float
            The upper boundary curve in y (same requirements as `gfun`).
        qfun : function or float
            The lower boundary surface in z.  It must be a function that takes
            two floats in the order (x, y) and returns a float or a float
            indicating a constant boundary surface.
        rfun : function or float
            The upper boundary surface in z. (Same requirements as `qfun`.)
        args : tuple, optional
            Extra arguments to pass to `func`.
        epsabs : float, optional
            Absolute tolerance passed directly to the innermost 1-D quadrature
            integration. Default is 1.49e-8.
        epsrel : float, optional
            Relative tolerance of the innermost 1-D integrals. Default is 1.49e-8.
        
        Returns
        -------
        y : float
            The resultant integral.
        abserr : float
            An estimate of the error.
        
        See Also
        --------
        quad : Adaptive quadrature using QUADPACK
        fixed_quad : Fixed-order Gaussian quadrature
        dblquad : Double integrals
        nquad : N-dimensional integrals
        romb : Integrators for sampled data
        simpson : Integrators for sampled data
        scipy.special : For coefficients and roots of orthogonal polynomials
        
        Notes
        -----
        For valid results, the integral must converge; behavior for divergent
        integrals is not guaranteed.
        
        **Details of QUADPACK level routines**
        
        `quad` calls routines from the FORTRAN library QUADPACK. This section
        provides details on the conditions for each routine to be called and a
        short description of each routine. For each level of integration, ``qagse``
        is used for finite limits or ``qagie`` is used, if either limit (or both!)
        are infinite. The following provides a short description from [1]_ for each
        routine.
        
        qagse
            is an integrator based on globally adaptive interval
            subdivision in connection with extrapolation, which will
            eliminate the effects of integrand singularities of
            several types.
        qagie
            handles integration over infinite intervals. The infinite range is
            mapped onto a finite interval and subsequently the same strategy as
            in ``QAGS`` is applied.
        """
