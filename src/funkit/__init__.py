"""
MathFlow - A Pythonic Interface for Symbolic and Numerical Mathematics
=====================================================================

MathFlow provides an object-oriented wrapper around SymPy with seamless integration of numerical
computations, making symbolic mathematics more intuitive and powerful for Python developers.

Overview
--------
MathFlow bridges the gap between symbolic mathematics (SymPy) and numerical computations (NumPy/SciPy),
offering a unified interface that maintains mathematical rigor while providing practical tools for
real-world problems. It extends SymPy's capabilities with features like operative closure, mutability
control, and automatic type conversions between different mathematical representations.

Key Features
------------
1. **Operative Closure**: Mathematical operations return new Expression objects by default, maintaining
   a functional programming style while preserving the original expressions.

2. **Mutability Control**: Choose between immutable (default) and mutable expressions for different
   use cases. Mutable expressions support in-place operations (+=, -=, etc.).

3. **Seamless Numerical Integration**: Every symbolic expression has a `.n` attribute providing
   numerical methods (integration, optimization, root finding) without manual lambdification.

4. **Enhanced Printing**: Flexible output formatting through the `.print` attribute supporting
   LaTeX, pretty printing, code generation, and more.

5. **Signal System**: Qt-like signals for tracking expression mutations and clones, enabling
   reactive programming patterns.

6. **Automatic Type Conversions**: Seamlessly switch between Poly and Expr representations based
   on context, accessing specialized methods from both.

Core Classes
------------
BaseExpression
    The foundation class providing the core interface. Delegates method calls to the internal
    SymPy expression while adding features like operative closure and mutability control.

Expression
    The primary user-facing class for general symbolic expressions. Includes numerical (.n) and
    printing (.print) proxies for enhanced functionality.

Polynomial
    Specialized Expression subclass that automatically maintains polynomial form, providing
    access to polynomial-specific methods like root finding and factorization.

Rational
    Represents rational functions (quotients of polynomials) with convenient access to numerator
    and denominator as Expression objects.

Function
    Alias for Expression, provided for semantic clarity when working with mathematical functions.

Design Philosophy
-----------------
MathFlow follows several key principles:

1. **Intuitive API**: Mathematical operations should feel natural in Python while maintaining
   mathematical correctness.

2. **Performance**: Automatic optimizations like Horner's method for polynomial evaluation and
   efficient numerical algorithms.

3. **Flexibility**: Support multiple workflows - functional (immutable) and imperative (mutable),
   symbolic and numerical.

4. **Extensibility**: Easy to extend with custom methods and integrate with other libraries.

5. **Type Safety**: Full type hints for better IDE support and catch errors early.

Dependencies
------------
- SymPy: Symbolic mathematics engine
- NumPy: Numerical array operations  
- SciPy: Advanced numerical algorithms (via bridge modules)
- Custom Internal modules: Specialized implementations for Padé approximations, parsing, etc.

Notes
-----
- The operative closure feature means operations return new objects by default. Use mutable=True
  for in-place operations.
- The .n attribute provides lazy lambdification - the expression is only compiled to a numerical
  function when first called.
- Type checking is fully supported with comprehensive type hints throughout.
- The signal system enables reactive programming patterns but is disabled by default for performance.

Useful Sympy Functions
----------------------
- Basic.doit()
- sympify()
- UnevaluatedExpr()  # makes an expression remain unevaluated or simplified until doit() is called.
- atoms()
- from sympy.abc import x, y, x
- x1, x2 = symbols('x1 x2')
- Basic.free_symbols  # property gives us the free symbols of an expression
"""
from core import Expression, Polynomial, Rational, Function
from sympy import abc
