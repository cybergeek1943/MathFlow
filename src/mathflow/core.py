# Import full libs for specialized cases
import sympy
import numpy as np

# Import specific sympy types
from sympy import Basic, Expr, Poly, Symbol, Number, Mul, Add

# Import sympy functions
from sympy import expand_mul, lambdify, horner, sympify, nsolve

# Import specific variables/symbols
from sympy.abc import x

# Import bridges
from mathflow import _repr_bridge_module
from mathflow import _numerical_bridge_module
from mathflow import _symbolic_bridge_module

# Import custom functionality
from mathflow import custom_numerical_funcs as cnf
from mathflow import custom_pade_implementation as cpi
from mathflow.parser import parse_expression_str

# Import helper functions
from copy import copy, deepcopy

# Import type checking tools and helpers
from functools import wraps, update_wrapper  # both do the same thing: wraps is a decorator, and update_wrapper is a normal function.
from typing import ParamSpec, TypeVar, TYPE_CHECKING, Union, Literal, Any, Self, Callable, Sequence
P = ParamSpec('P')  # Captures parameter signature
T = TypeVar('T')    # Captures return type
SympyExpr = Expr | Poly  # the reason that Poly does not inherit from Expr is for representation and performance reasons. In true mathematical taxonomy form, Poly would inherit from Expr. Hence, we use the 'Union' of their types so that our proxy classes are compatible with both.


# Helper Classes
class DefaultArgs:
    __slots__ = 'args', 'kwargs'
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args: tuple[Any, ...] = args
        self.kwargs: dict[str, Any] = kwargs

    def __bool__(self) -> bool:
        return bool(self.args) or bool(self.kwargs)


class Signal:
    """Implements a QT signal-like system using traditional callbacks."""
    __slots__ = 'callables', 'callables_count', 'disabled'
    def __init__(self) -> None:
        self.callables: list[Callable] = []
        self.callables_count: int = 0
        self.disabled: bool = True

    def __deepcopy__(self, memo) -> 'Signal':  # implement this so that cloning works in the BaseExpression
        n: Signal = copy(self)
        n.callables = self.callables.copy()
        return n

    def emit(self, *args, **kwargs) -> None:
        if self.disabled: return
        for c in self.callables: c(*args, **kwargs)

    def connect(self, func: Callable) -> None:
        if func in self.callables: return
        self.callables.append(func)
        self.__increment_callables_count__(1)

    def disconnect(self, func: Callable) -> None:
        self.callables.remove(func)
        self.__increment_callables_count__(-1)

    def __increment_callables_count__(self, n: int) -> None:
        self.callables_count += n
        self.disabled = self.callables_count == 0


# Base Classes
if TYPE_CHECKING:  # Useful for the IDE or Type checkers to see
    from mathflow._symbolic_bridge_stub import SympyModule
    class _BaseExpression(Expr, Poly, SympyModule):
        pass  # If we want to customize how one of the methods of Expr, Poly, or BridgeClass look, we can override it here.
else:
    _BaseExpression = object


class BaseExpression(_BaseExpression):
    """
    Base Expression Interface... subclassed by specific interfaces such as Expression, Polynomial, Function, RationalFunction, etc.
    Basic function that interact with the internal Expr should be defined here.
    """
    if TYPE_CHECKING:  # for some reason, the type checking thinks Expr or Poly have a .expr property that needs a getter and setter
        @property
        def expr(self):
            return self._expr
        @expr.setter
        def expr(self, value):
            self._expr = value

    @property
    def expr_type(self) -> type[Self]:
        return type(self.expr)

    def as_sympy_expr(self) -> Expr:
        return self.expr

    def set_expr(self, expr: Union[SympyExpr, 'BaseExpression'] | str | Any, **kwargs) -> None:
        """If `expr` is of type Any, *args and **kwargs are used as arguments to sympy.sympify()."""
        if expr is None:
            self.expr = sympify(0)
        elif isinstance(expr, SympyExpr):
            self.expr = expr
        elif isinstance(expr, str):
            self.expr = parse_expression_str(expr, **kwargs)
        elif isinstance(expr, BaseExpression):
            self.expr = expr.expr
        else:
            try: self.expr = sympify(expr, **kwargs)
            except: self.expr = expr

        # emit signal if expr is changed.
        if hasattr(self, 'on_self_mutated'):
            self.on_self_mutated.emit(self)

    def __init__(self, expr: Union[SympyExpr, 'BaseExpression'] | str | Any = None,
                 operative_closure: bool = True,
                 mutable: bool = False,
                 auto_handle_poly_switching: bool = True,
                 **kwargs) -> None:
        # Behavioral Attributes
        self.operative_closure: bool = operative_closure
        self.mutable: bool = mutable
        self.auto_handle_poly_switching: bool = auto_handle_poly_switching

        # Internal Representation
        self.expr: SympyExpr | Any = ...  # self.expr strongly prefers to be of type SympyExpr = Union[Expr, Poly].
        self.set_expr(expr, **kwargs)

        # Signals
        self.on_self_mutated: Signal = Signal()
        self.on_self_cloned: Signal = Signal()

    def __deepcopy__(self, memo) -> 'BaseExpression':  # we must implement this so that __getattr__ does not cause problems with the deepcopy() function.
        n: BaseExpression = object.__new__(self.__class__)
        for k, v in self.__dict__.items():
            setattr(n, k, deepcopy(v, memo))
        return n

    def __hash__(self):  # to avoid errors in hash based collections when mutable is enabled.
        if self.mutable:
            raise TypeError(f"unhashable type: '{self.__class__.__name__}' (mutable=True)")
        return hash(self.expr)

    def _mutate_self(self, new_expr: SympyExpr) -> Self:
        self.expr = new_expr
        self.on_self_mutated.emit(self)
        return self  # return self for chaining purposes

    def _clone_self(self, new_expr: SympyExpr) -> 'BaseExpression':
        temp_pointer: SympyExpr = self.expr
        self.expr = None  # we don't want to copy the expr
        n: BaseExpression = deepcopy(self)
        n.expr = new_expr
        n.on_self_cloned.emit(n)
        self.expr = temp_pointer
        return n

    # Now delegate everything to expr which has full sympy functionality
    def __getattr__(self, name: str) -> Union[Self, 'BaseExpression'] | Any:
        try:
            attr: Any = getattr(self.expr, name)
        except:  # auto switch between Poly and Expr depending on context of `name`
            if self.auto_handle_poly_switching:
                if hasattr(Poly, name) and isinstance(self.expr, Expr):
                    self.expr: Poly = self.expr.as_poly()
                    attr: Any = getattr(self.expr, name)  # we could do a recursive call instead, but that could result in recursion errors in the event of unsuccessful conversion.
                elif self.expr.is_Poly and hasattr(Expr, name):
                    self.expr: Expr = self.expr.as_expr()
                    attr: Any = getattr(self.expr, name)
            raise

        if self.operative_closure and callable(attr):
            def wrapper(*args, **kwargs) -> Any:
                result: Any = attr(*args, **kwargs)
                if isinstance(result, SympyExpr):
                    if self.mutable:  # IMPORTANT: checking mutable must be before checking operative_closure!
                        return self._mutate_self(new_expr=result)
                    if self.operative_closure:
                        return self._clone_self(new_expr=result)
                return result
            update_wrapper(wrapper, attr)  # make the wrapper of attr look like the attr itself for introspection purposes.
            return wrapper
        return attr  # Properties and non-callables pass through directly

    def enable_polynomial_tools(self, b: bool, *gens: Symbol) -> bool:
        """Because of backend implementation differences between Poly and Expr (for performance reasons, even though the mathematical taxonomy is broken), having the option to convert the internal expr to these forms (and give us the specific tools) is valuable."""
        if b and isinstance(self.expr, Expr) and self.expr.is_polynomial(*gens):
            self.expr = self.expr.as_poly(*gens)
            return True
        elif not b and isinstance(self.expr, Poly):
            self.expr = self.expr.as_expr()
            return True
        return False

    def is_polynomial(self, *syms) -> bool:
        """Because Poly and Expr are essentially merged, this provides a universal method to check if the expr is a polynomial."""
        return self.expr.is_Poly or self.expr.is_polynomial(*syms)

    @property
    def free_symbols_ordered(self) -> list[Basic]:
        return sorted(self.expr.free_symbols, key=lambda s: (  # order the symbols according to mathematical convention
            0 if s.name in 'xyz' else  # Variables first (x, y, z)
            1 if s.name in 'tuv' else  # Then t, u, v (often time/indices)
            2,  # Parameters last (a, b, c, etc.)
            s.name  # Secondary sort: alphabetical within each group
        ))

    def lambdify(self, *args, **kwargs) -> Callable:
        expr: SympyExpr = self.expr
        try:  # if expr is in polynomial form, try to use horner's method on it to optimize its evaluation.
            if self.is_polynomial():
                expr = horner(expr)
        except: pass

        # if there are positional arguments
        if args:
            return lambdify(args[0], expr, *args[1:], **kwargs)
        # if sympy.lambdify()'s first parameter 'args' is defined in the kwargs of this method, place it where it belongs.
        if 'args' in kwargs:
            _ = kwargs['args']; kwargs.pop('args')
            return lambdify(_, expr, **kwargs)
        # if the first two arguments of sympy.lambdify() are not in this methods arguments, assume defaults and pass any additional kwargs.
        return lambdify(self.free_symbols_ordered, expr, **kwargs)

    def __str__(self) -> str:
        return self.expr.__str__()

    def __repr__(self) -> str:
        return self.expr.__repr__()


def _handle_args_with_mathflow_objects(args, kwargs) -> tuple[list, dict]:
    out_args = [arg.expr if isinstance(arg, BaseExpression) else arg for arg in args]
    for k, v in kwargs.items():
        if isinstance(v, BaseExpression):
            kwargs[k] = v.expr
    return out_args, kwargs


def _delegate_dunder_method(name: str) -> Callable:
    """This is a function that helps delegate dunder methods to BaseExpression.expr object."""
    def dunder_method(self: BaseExpression, *args, **kwargs) -> Any:
        args, kwargs = _handle_args_with_mathflow_objects(args, kwargs)  # this enables operation between types BaseExpression and BaseExpression.
        result: Any = getattr(self.expr, name)(*args, **kwargs)
        if self.operative_closure and isinstance(result, SympyExpr):
            return self._clone_self(new_expr=result)
        return result
    update_wrapper(dunder_method, getattr(Expr, name))  # make the operator_method wrapper look like the Expr methods being wrapped (for introspection purposes). Same as using @functools.wraps() decorator.
    return dunder_method


def _delegate_in_place_dunder_method(name: str) -> Callable:
    """This is a function that helps delegate in-place dunder methods to BaseExpression object."""
    def in_place_dunder_method(self: BaseExpression, other: Any) -> Union[BaseExpression, NotImplemented]:
        if not self.mutable:
            return NotImplemented  # the normal non in-place dunder method will be used as a fallback with traditional assignment. Thus, the id() of the identifier will change. This is default python behavior (returning NotImplemented is the same as if the method was never defined).
        if isinstance(other, BaseExpression):
            other: SympyExpr = other.expr
        return self._mutate_self(new_expr=getattr(self.expr, name.replace("__i", "__"))(other))
    in_place_dunder_method.__name__ = name
    return in_place_dunder_method


def _setup_BaseExpression_dunder_methods() -> None:
    """Only run this function once... after the BaseExpression has been defined.

    This is necessary because dunder methods don't go through the usual .__getattr__() or .__getattribute__() methods:
    https://docs.python.org/3/reference/datamodel.html#object.__getattribute__
    """
    math_operations: set[str] = {
        '__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__',
        '__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__',
        '__mod__', '__rmod__', '__pow__', '__rpow__', '__matmul__', '__rmatmul__',
        '__neg__', '__pos__', '__abs__', '__divmod__', '__rdivmod__'
    }

    # Comparison operations
    comparison_ops: set[str] = {
        '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
    }

    # Bitwise operations (if needed for SymPy)
    bitwise_ops: set[str] = {
        '__and__', '__rand__', '__or__', '__ror__', '__xor__', '__rxor__',
        '__lshift__', '__rlshift__', '__rshift__', '__rrshift__', '__invert__',
    }

    # Conversion operations
    conversion_ops: set[str] = {
        '__int__', '__float__', '__complex__', '__bool__',
        # '__str__', '__repr__', '__format__',  # Skip these
    }

    # Container operations (if SymPy uses them)
    container_ops: set[str] = {
        '__len__', '__getitem__', '__setitem__', '__delitem__',
        '__contains__', '__iter__',
    }

    misc_ops: set[str] = {
        '__call__',  # Function application
        '__round__', '__trunc__', '__floor__', '__ceil__',  # Rounding operations
        '__index__',  # For integer conversion in contexts requiring integers
        '__bytes__',  # If SymPy supports byte conversion
        '__enter__',
        '__exit__'
    }

    delegated_methods: set[str] = math_operations | comparison_ops | bitwise_ops | conversion_ops | container_ops | misc_ops
    for attr_name in delegated_methods:
        if not hasattr(Expr, attr_name) or hasattr(BaseExpression, attr_name):
            continue
        setattr(BaseExpression, attr_name, _delegate_dunder_method(attr_name))

    # Add in-place operation dunders
    in_place_operators: set[str] = {
        '__iadd__',  # +=
        '__isub__',  # -=
        '__ifloordiv__',  # //=
        '__itruediv__',  # /=
        '__imod__',  # %=
        '__imul__',  # *=
        '__ipow__',  # **=
        '__imatmul__',  # @=
        '__iand__',  # &=
        '__ior__',  # |=
        '__ixor__',  # ^=
        '__irshift__',  # >>=
        '__ilshift__',  # <<=
    }
    for dunder_name in in_place_operators:
        if hasattr(BaseExpression, dunder_name):
            continue
        setattr(BaseExpression, dunder_name, _delegate_in_place_dunder_method(dunder_name))
_setup_BaseExpression_dunder_methods()


# ================================ Full Implementation ================================
if TYPE_CHECKING:  # Useful for the IDE or Type checkers to see
    from mathflow._repr_bridge_stub import SympyReprModule
    _BasePrintProxy = SympyReprModule
else:
    _BasePrintProxy = object


class PrintProxy(_BasePrintProxy):
    """OO Interface for string rendering of expressions."""
    __slots__ = 'expr', 'default_method', 'default_args', 'repr_same_as_str'
    method_types = Literal['normal', 'pretty', 'latex', 'mathml', 'mathematica_code', 'ccode', 'cxxcode', 'rcode', 'fcode', 'smtlib_code', 'maple_code', 'jscode', 'julia_code', 'octave_code', 'rust_code', 'pycode', 'tree'] | str

    def __init__(self, expr: SympyExpr) -> None:
        self.expr: SympyExpr = expr
        self.default_method: PrintProxy.method_types = 'normal'
        self.default_args: DefaultArgs = DefaultArgs()
        self.repr_same_as_str: bool = True  # if we want the __repr__() to output the same str as __str__()

    def __deepcopy__(self, memo) -> 'PrintProxy':  # we want to avoid copying self.expr (we do, however, want to deepcopy self.default_args)
        n: PrintProxy = copy(self)
        n.default_args = deepcopy(self.default_args)
        return n

    def __str__(self) -> str:
        return str(self.expr) if self.default_method == 'normal' else getattr(self, self.default_method)()

    def __repr__(self) -> str:
        return str(self) if self.repr_same_as_str else repr(self.expr)

    def __call__(self, method: 'PrintProxy.method_types' = None, *args, **kwargs) -> None:
        if method is None:
            print(self)
            return
        print(getattr(self, method)(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        item: Any = _repr_bridge_module.get_attr(name)
        if not callable(item):
            return item  # non-callables pass through directly

        def wrapper(*args, **kwargs) -> Any:
            return item(self.expr,
                        *(args if args else self.default_args.args),
                        **(kwargs if kwargs else self.default_args.kwargs))

        update_wrapper(wrapper, item)  # make the wrapper of item look like the attr itself for introspection purposes.
        return wrapper


if TYPE_CHECKING:  # Useful for the IDE or Type checkers to see
    from mathflow._numerical_bridge_stub import ScipyModule
    _BaseNumericalProxy = ScipyModule
else:
    _BaseNumericalProxy = object


class NumericalProxy(_BaseNumericalProxy):
    """Only provide the bare minimum needed for numerical methods. (Explain the design philosophy later.)"""
    __slots__ = 'ref', '_f', 'params', 'default_lambdify_args'

    def __init__(self, expr: Union['Expression', None]) -> None:
        self.ref: Expression | None = expr  # allow for None type in case we want this Proxy to have operative closure on numerical functions that return callables.
        self._f: Callable | None = None
        self.params: DefaultArgs = DefaultArgs()
        self.default_lambdify_args: DefaultArgs = DefaultArgs()

    def __call__(self, *args, **kwargs) -> Any:
        if not self._f and self.ref:
            self._f = self.ref.lambdify(*self.default_lambdify_args.args, **self.default_lambdify_args.kwargs)
        if self.params:
            args += self.params.args
            kwargs = {**self.params.kwargs, **kwargs}
        try:
            return self._f(*args, **kwargs)
        except Exception as e:  # in case we need to vectorize _f
            try:
                vf: Callable = np.vectorize(self._f)
                return vf(*args, **kwargs)
            except:
                raise e

    def __deepcopy__(self, memo):
        n: NumericalProxy = copy(self)
        n.default_lambdify_args = deepcopy(self.default_lambdify_args)
        return n

    def __getattr__(self, name: str) -> Any:
        item: Any = _numerical_bridge_module.get_attr(name)
        if not callable(item):
            return item

        def wrapper(*args, **kwargs) -> Any:
            return item(self, *args, **kwargs)

        update_wrapper(wrapper, item)  # make the wrapper of item look like the attr itself for introspection purposes.
        return wrapper

    def __str__(self):
        if self._f is None:
            return f"NumericalProxy({self.ref if self.ref else 'None'}) [not lambdified yet]"
        import inspect
        try: code: str = inspect.getsource(self._f)
        except: code: str = "Could not retrieve the lambdified function's code"
        return f"NumericalProxy({str(self._f)}) [source code below]\n{code}"

    def __repr__(self):
        return repr(self._f)

    # ==== Modified Aliases (unmodified aliases are defined in the respective bridge module) ====
    def integrate(self, *args, method: str = 'quad', **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def optimize(self, *args, method: str = 'minimize', **kwargs):
        return getattr(self, method)(*args, **kwargs)

    # ==== Custom ====
    def derivative_lambda(self, *args, df_order: int = 1, **kwargs) -> 'NumericalProxy':
        """Returns a lambda function which is the numerical derivative of self."""
        new_proxy: NumericalProxy = NumericalProxy(None)
        new_proxy._f = cnf.derivative_lambda(self, *args, df_order, **kwargs)
        return new_proxy

    def all_roots(self,
                  args: tuple = (),
                  x0s: Sequence[float] = None,
                  bounds: tuple[float, float] = (-10, 10),
                  resolution: int = 100,
                  dup_tol: float = 1e-4,
                  method: str = 'hybr',
                  tol: float = 1e-8,
                  **kwargs) -> np.ndarray:
        return cnf.all_roots(self, args, x0s, bounds, resolution, dup_tol, method, tol, **kwargs)

    solve_all = all_roots  # add alias for naming consistency

    def all_poly_roots(self, cleaning_tol: float = 1e-8, **kwargs) -> np.ndarray:
        if self.ref and self.ref.is_polynomial():
            return cnf.all_poly_roots(self.ref.all_coeffs(), cleaning_tol)
        best_args: dict = cnf.best_kwargs_for_polynomials_in_all_roots_func.copy()
        best_args.update(kwargs)
        return self.all_roots(**best_args)


def _closure_compatible_method(f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)  # make the wrapper of attr look like the attr itself for introspection purposes.
    def wrapper(self, *args, **kwargs) -> Any:
        result: Any = f(self, *args, **kwargs)
        if isinstance(result, SympyExpr):
            if self.mutable:  # IMPORTANT: checking mutable must be before checking operative_closure!
                return self._mutate_self(new_expr=result)
            if self.operative_closure:
                return self._clone_self(new_expr=result)
        return result
    return wrapper


class Expression(BaseExpression):
    def __init__(self, expr: Union[SympyExpr, 'BaseExpression'] | str | Any = None, **kwargs) -> None:
        super().__init__(expr, **kwargs)  # after some already defined attributes so that kwargs can be used to create or set attributes.
        self.n: NumericalProxy = NumericalProxy(self)
        self.print: PrintProxy = PrintProxy(self.expr)

        # Connect to signals so that the lambdified_expr stays up to date with internal expression (weather mutable or immutable).
        self.on_self_mutated.connect(Expression.__sync_expr)
        self.on_self_cloned.connect(Expression.__sync_expr)

    @staticmethod
    def __sync_expr(self: 'Expression'):  # This is a static function so that when Expression is cloned, the signals are still directed to the right method in memory (the cloned self is emitted as an arg).
        self.print.expr = self.expr
        self.n.ref = self
        self.n._f = None

    def __call__(self, *args, **kwargs) -> Any:
        return self.n(*args, **kwargs)

    def __str__(self) -> str:
        return str(self.print)

    def __repr__(self) -> str:
        return repr(self.print)

    # Now delegate everything to expr which has full sympy functionality
    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)  # propagate request to BaseExpression
        except:  # try to get function from the bridge module
            attr: Any = _symbolic_bridge_module.get_attr(name)

        if callable(attr):  # don't check for operative_closure because attr needs self.expr passed to it (attr must always be wrapped).
            def wrapper(*args, **kwargs) -> Any:
                result: Any = attr(self.expr, *args, **kwargs)
                if isinstance(result, SympyExpr):
                    if self.mutable:  # IMPORTANT: checking mutable must be before checking operative_closure!
                        return self._mutate_self(new_expr=result)
                    if self.operative_closure:
                        return self._clone_self(new_expr=result)
                return result
            update_wrapper(wrapper, attr)  # make the wrapper of attr look like the attr itself for introspection purposes.
            return wrapper

        return attr  # Properties and non-callables pass through directly

    # ======================== Custom Implementations ========================
    # noinspection PyDefaultArgument
    def nsolve_all(self,
                   syms: Sequence[Symbol] | None = None,
                   param_values: tuple[Any] = (),
                   x0s: Sequence[float] = None,
                   bounds=(-10, 10),
                   resolution: int = 100,
                   dup_tol: float = 1e-8,
                   _machine_args: dict = {},
                   **kwargs) -> list:
        """Find all roots numerically with variable precision"""
        if not x0s:
            # Use machine precision to get reasonable starting points
            x0s = self.n.all_roots(param_values,
                                   x0s,
                                   bounds,
                                   resolution,
                                   **_machine_args)
        high_precision_roots: list = []
        for x in x0s:
            try:
                root = nsolve(self.expr, syms, x, **kwargs)
                if any(abs(root - r) < dup_tol for r in high_precision_roots):
                    continue  # Skip duplicates within tolerance
                high_precision_roots.append(root)
            except:
                continue  # Skip if nsolve fails for this guess

        return high_precision_roots

    @_closure_compatible_method
    def pade(self,
         m: int,
         n: int = None,
         x0: float = 0,
         x: Symbol = None,
         return_type: Literal['rational', 'pair', 'coeffs', 'dict'] = 'rational',
         expand_terms: bool = True,
         clear_coeff_denominators: bool = True,
         rationalize_mpmath_coeffs: bool = True,
         backend: Literal['auto', 'symbolic', 'verbose', 'mpmath'] = 'auto') -> Union['Expression', Expr] | tuple[Expr, Expr] | tuple[Sequence[Number], Sequence[Number]] | dict:  # TODO specify return types
        if x is None:
            x: Basic = self.free_symbols_ordered[0]
        return cpi.pade(self.expr, x, x0, m, n if n else m,
                        return_type,
                        expand_terms,
                        clear_coeff_denominators,
                        rationalize_mpmath_coeffs,
                        backend,
                        _lambdified_expr=self.n)

    def ai(self) -> None:
        # TODO 👉👉👉 Work on an Mathstral AI backend.
        raise NotImplementedError('This method has not been developed yet...')

    # ======================== Alias Methods ========================
    def partial_fractions(self, x, full: bool = False, **options) -> Union['Expression', SympyExpr]:
        return self.apart(x, full=full, **options)

    # ======================== Factory Methods ========================
    @classmethod
    def from_coeffs(cls, coeffs: Sequence[Number], var: Symbol = x, ascending: bool = True, **kwargs) -> 'Expression':
        if not coeffs:
            raise ValueError('At least one coeff must be provided')
        if not ascending:
            coeffs = coeffs[::-1]  # Reverse for highest-first
        return cls(Add(*[c*var**i for i, c in enumerate(coeffs)]), **kwargs)

    @classmethod
    def from_roots(cls, roots: Sequence[Number], var: Symbol = x, **kwargs) -> 'Expression':
        if not roots:
            raise ValueError('At least one root must be provided')
        return cls(expand_mul(Mul(*[var - r for r in roots])), **kwargs)


class Polynomial(Expression):
    def __init__(self, expr: Union[SympyExpr, 'BaseExpression'] | str | Any = None, **kwargs) -> None:
        super().__init__(expr, **kwargs)
        self.enable_polynomial_tools(True)

    # NOTE: we must define class methods because the delegator only delegates normal methods
    @classmethod
    def from_list(cls, rep, *gens, **args) -> 'Polynomial':
        """Construct a polynomial from a ``list``. """
        if not gens:
            gens = (x,)
        return cls(Poly.from_list(rep, *gens, **args))

    @classmethod
    def from_dict(cls, rep, *gens, **args) -> 'Polynomial':
        """Construct a polynomial from a ``dict``. """
        return cls(Poly.from_dict(rep, *gens, **args))

    @classmethod
    def from_poly(cls, rep, *gens, **args) -> 'Polynomial':
        """Construct a polynomial from a polynomial. """
        return cls(Poly.from_poly(rep, *gens, **args))

    @classmethod
    def from_expr(cls, rep, *gens, **args) -> 'Polynomial':
        """Construct a polynomial from an expression. """
        return cls(Poly.from_expr(rep, *gens, **args))


class Rational(Expression):
    def __init__(self, expr: Union[SympyExpr, 'BaseExpression'] | str | Any = None, **kwargs) -> None:
        super().__init__(expr, **kwargs)
        self._p, self._q = self.expr.as_numer_denom()
        self.on_self_mutated.connect(self.__sync_expr)
        self.on_self_cloned.connect(self.__sync_expr)

    @staticmethod
    def __sync_expr(self: 'Rational'):  # is not actually overriding parent method because this is static
        self._p, self._q = self.expr.as_numer_denom()

    @property
    def numerator(self) -> Expression | SympyExpr:
        return Expression(self._p) if self.operative_closure else self._p

    @property
    def denominator(self) -> Expression | SympyExpr:
        return Expression(self._q) if self.operative_closure else self._q

    @numerator.setter
    def numerator(self, value: SympyExpr | Expression) -> None:
        if isinstance(value, Expression):
            value: SympyExpr = value.expr
        self.set_expr(value / self._q)  # set_expr() automatically emits the mutation signal

    @denominator.setter
    def denominator(self, value: SympyExpr | Expression) -> None:
        if isinstance(value, Expression):
            value: SympyExpr = value.expr
        self.set_expr(self._p / value)  # set_expr() automatically emits the mutation signal

    # noinspection PyMethodOverriding
    @classmethod
    def from_coeffs(cls, p: Sequence[Number], q: Sequence[Number], var: Symbol = x, ascending: bool = True, **kwargs) -> 'Rational':
        if not p or not q:
            raise ValueError('At least one coeff must be provided for both p and q')
        if not ascending:
            p, q = p[::-1], q[::-1]  # Reverse for highest-first
        return cls(Add(*[c*var**i for i, c in enumerate(p)]) / Add(*[c*var**i for i, c in enumerate(q)]), **kwargs)


# Aliases:
class Function(Expression):
    """Alias to Expression (for contextual naming clarity)."""
    pass


if __name__ == '__main__':
    from sympy.abc import x, z
    f = Rational.from_coeffs([1, 2], [3, 4, 5])
    print(f.on_self_mutated.callables)
