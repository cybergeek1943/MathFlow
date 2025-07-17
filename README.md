# FunKit 🧮

**A Pythonic Interface for Symbolic and Numerical Mathematics**

FunKit bridges the gap between symbolic mathematics (SymPy) and numerical computations (NumPy/SciPy), offering a unified interface that maintains mathematical rigor while providing practical tools for real-world problems.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- **🔒 Operative Closure**: Mathematical operations return new Expression objects by default, maintaining functional programming principles
- **⚡ Mutability Control**: Choose between immutable (default) and mutable expressions for different workflows
- **🔗 Seamless Numerical Integration**: Every symbolic expression has a `.n` attribute providing numerical methods without manual lambdification
- **🎨 Enhanced Printing**: Flexible output formatting through the `.print` attribute (LaTeX, pretty printing, code generation)
- **📡 Signal System**: Qt-like signals for tracking expression mutations and clones, enabling reactive programming
- **🔄 Automatic Type Conversions**: Seamlessly switch between Poly and Expr representations based on context

## 🚀 Quick Start

```python
from funkit import Expression, Polynomial, Rational
from sympy.abc import x, y

# Create expressions naturally
f = Expression("2x^2 + 3x + 1")
g = Expression("sin(x) + cos(x)")

# Automatic operative closure - operations return new objects
h = f + g  # f and g remain unchanged
print(f"f: {f}")
print(f"g: {g}")  
print(f"h: {h}")

# Numerical evaluation made easy
result = f.n(2.5)  # Evaluate at x = 2.5
print(f"f(2.5) = {result}")

# Beautiful printing options
f.print.latex()     # LaTeX output
f.print.pretty()    # Pretty ASCII art
f.print.ccode()     # C code generation
```

## 📚 Core Classes

### Expression
The primary class for general symbolic expressions with numerical and printing capabilities.

```python
# Create from string with natural notation
expr = Expression("2x^2 + ln(|x-1|)")

# Or from SymPy objects
from sympy import sin, cos
expr = Expression(sin(x) + cos(x))

# Numerical methods via .n attribute
roots = expr.n.all_roots(bounds=(-10, 10))
integral = expr.n.quad(0, 1)  # Numerical integration

# Symbolic operations
derivative = expr.diff(x)
expanded = expr.expand()
```

### Polynomial
Specialized Expression subclass with polynomial-specific functionality.

```python
# Create from coefficients (ascending order by default)
poly = Polynomial.from_coeffs([1, 2, 3])  # 1 + 2x + 3x²

# Create from roots
poly = Polynomial.from_roots([1, 2, 3])   # (x-1)(x-2)(x-3)

# Access polynomial-specific methods
coeffs = poly.all_coeffs()
degree = poly.degree()
roots = poly.n.all_poly_roots()  # Optimized polynomial root finding
```

### Rational
For rational functions (quotients of polynomials).

```python
# Create from numerator and denominator coefficients
rational = Rational.from_coeffs([1, 2], [1, 1, 1])  # (1 + 2x)/(1 + x + x²)

# Access numerator and denominator as Expression objects
num = rational.numerator
den = rational.denominator

# Partial fraction decomposition
pf = rational.partial_fractions(x)
```

## 🔧 Advanced Features

### Mutability Control

```python
# Immutable (default) - functional style
f = Expression("x^2")
g = f + 1  # f unchanged, g is new object

# Mutable - imperative style  
f = Expression("x^2", mutable=True)
f += 1     # f is modified in-place
```

### Padé Approximations

```python
# High-quality Padé approximants for function approximation
f = Expression("exp(x)")
pade_approx = f.pade(m=3, n=3, x0=0)  # [3/3] Padé approximant around x=0

# Multiple backends available
pade_fast = f.pade(3, 3, backend='mpmath')      # Fast numerical
pade_exact = f.pade(3, 3, backend='symbolic')   # Exact symbolic
pade_verbose = f.pade(3, 3, backend='verbose')  # Educational output
```

### Flexible String Parsing

```python
# Natural mathematical notation
expr = Expression("2x^2 + ln(|x-1|)")   # Powers, natural log, absolute value

# LaTeX input support  
expr = Expression(r"\frac{x^2+1}{x-1}")  # LaTeX fractions

# Implicit multiplication
expr = Expression("2x sin(x)")           # Automatically becomes 2*x*sin(x)
```

### Signal System for Reactive Programming

```python
def on_change(expr):
    print(f"Expression changed to: {expr}")

f = Expression("x^2")
f.on_self_mutated.connect(on_change)
f.on_self_cloned.connect(on_change)

# Signals disabled by default for performance
f.on_self_mutated.disabled = False
```

## 🎯 Numerical Computing

FunKit excels at bridging symbolic and numerical mathematics:

```python
f = Expression("x^3 - 2*x^2 + x - 1")

# Root finding
all_roots = f.n.all_roots(bounds=(-5, 5))
specific_root = f.nsolve_all(x0s=[1.0])  # High-precision solve

# Numerical calculus
derivative_func = f.n.derivative_lambda(df_order=2)  # 2nd derivative function
integral_result = f.n.integrate(-1, 1)              # Definite integral

# Optimization
minimum = f.n.minimize(bounds=[(-2, 2)])
```

## 📋 Dependencies

- **SymPy**: Symbolic mathematics engine
- **NumPy**: Numerical array operations  
- **SciPy**: Advanced numerical algorithms
- **mpmath**: High-precision arithmetic (for Padé approximations)
- **tabulate**: Pretty table printing (for verbose mode)

## 🏗️ Design Philosophy

FunKit follows several key principles:

1. **Intuitive API**: Mathematical operations should feel natural in Python
2. **Performance**: Automatic optimizations (Horner's method, efficient algorithms)
3. **Flexibility**: Support both functional and imperative programming styles
4. **Extensibility**: Easy integration with other mathematical libraries
5. **Type Safety**: Comprehensive type hints for better IDE support

## 📖 Examples

### Function Analysis
```python
# Analyze a complex function
f = Expression("sin(x) * exp(-x^2/2)")

# Symbolic analysis
critical_points = f.diff(x).solve(x)
taylor_series = f.series(x, 0, 6)

# Numerical analysis  
roots = f.n.all_roots(bounds=(-5, 5))
plot_points = [(x_val, f.n(x_val)) for x_val in np.linspace(-3, 3, 100)]
```

### Polynomial Operations
```python
# Create and manipulate polynomials
p1 = Polynomial.from_coeffs([1, 0, 1])      # x² + 1
p2 = Polynomial.from_roots([1, -1])         # (x-1)(x+1) = x² - 1

# Polynomial arithmetic
sum_poly = p1 + p2                          # 2x²
product = p1 * p2                           # (x²+1)(x²-1) = x⁴ - 1

# Advanced operations
factors = product.factor()
roots = product.n.all_poly_roots()
```

### Rational Function Analysis
```python
# Rational functions
rational = Rational("(x^2 + 1)/(x^2 - 1)")

# Decomposition
partial_fractions = rational.partial_fractions(x)
poles = rational.denominator.solve(x)

# Asymptotic analysis
limit_at_inf = rational.limit(x, float('inf'))
residues = [rational.residue(x, pole) for pole in poles]
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Code style and conventions
- Testing requirements  
- Documentation standards
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on the excellent [SymPy](https://www.sympy.org/) symbolic mathematics library
- Numerical computing powered by [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)
- High-precision arithmetic via [mpmath](http://mpmath.org/)

---

**Ready to revolutionize your mathematical computing workflow?** 

```bash
pip install funkit
```

Get started with the [documentation](docs/) and [examples](examples/) to see FunKit in action!