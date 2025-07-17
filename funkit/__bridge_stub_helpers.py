"""
Bridge Stub Generator for Type Checking

This module provides utilities for generating Python stub files (.pyi-style content)
from bridge modules that expose third-party library functions as methods on a proxy class.

Purpose
-------
When creating bridge/proxy classes that dynamically expose functions from libraries
like SymPy or SciPy as methods (typically via __getattr__), IDEs and type checkers
cannot infer the available methods or their signatures. This module generates stub
definitions that can be used in TYPE_CHECKING blocks to provide proper IDE support
and static type checking.

Key Features
------------
- Extracts function signatures and converts them to method signatures (replacing
  first parameter with 'self')
- Preserves type annotations from original functions
- Extracts and formats docstrings with configurable truncation
- Handles edge cases like functions with missing signatures or complex default values
- Generates complete stub files with import statements and class definitions
- Supports multiple bridge modules (symbolic, numerical, repr, etc.)

Typical Workflow
---------------
1. Create a bridge module that exposes library functions via __all__
2. Use generate_method_signatures() to extract method signatures from the bridge
3. Use create_stub() to generate a complete stub file
4. Import the generated class in TYPE_CHECKING blocks for IDE support

This approach enables clean separation between runtime bridge functionality and
static type information, providing the best of both worlds: dynamic method
exposure with full IDE and type checker support.
"""
from types import ModuleType, FunctionType
from typing import Sequence
import inspect


def get_func_docstring(func: FunctionType, truncate_at: str | tuple[str, ...] = None) -> str:
    docstring = func.__doc__
    if not docstring:  # Handle None case
        return ''
    if truncate_at:
        if isinstance(truncate_at, tuple):
            _pos = [pos for t in truncate_at if (pos := docstring.find(t)) != -1]
            pos = min(_pos) if _pos else -1
        else:
            pos: int = docstring.find(truncate_at)
        if pos != -1:  # Only truncate if found
            docstring = docstring[:pos]
    return docstring


def create_function_string(name: str, params: Sequence[str] | str = (), return_annotation: str = '', docstring: str = '') -> str:
    if params == 'var_args':
        params = ['self', '*args', '**kwargs']
    tab = '    '
    tab_2 = 2 * tab
    s: str = f'{tab}def {name}({', '.join(params)})'
    s += f' -> {return_annotation}:' if return_annotation else ':'
    s += f' ...' if not docstring else f'\n{tab_2}"""{docstring.replace('\n', f'\n{tab_2}')}"""'
    return s


def generate_method_signatures(module: ModuleType,
                               return_annotations: bool = True,
                               extract_docs: bool = True,
                               truncate_docs_at: str | tuple[str, ...] = None) -> str:
    """Generate method signatures for TYPE_CHECKING block."""
    output: list[str] = []
    for name in module.__all__:
        func: FunctionType = getattr(module, name)
        # Get the docstring
        docstring: str = '' if not extract_docs else get_func_docstring(func, truncate_docs_at)
        try:
            # Get the function signature
            sig = inspect.signature(func)

            # Convert parameters to string, replacing first parameter with 'self'
            params: list[inspect.Parameter] = list(sig.parameters.values())
            param_strings: list[str] = ['self']  # Replace first parameter with 'self'
            if params:
                if str(params[0]).startswith('*'):
                    param_strings.append(str(params[0]))
                for param in params[1:]:
                    param_strings.append(str(param))

            # Handle return annotation if present
            return_annotation: str = ''
            if return_annotations and sig.return_annotation != inspect.Signature.empty:
                if hasattr(sig.return_annotation, '__name__'):
                    return_annotation = sig.return_annotation.__name__
                else:
                    return_annotation = str(sig.return_annotation)

            output.append(create_function_string(name, param_strings, return_annotation, docstring))
        except:
            # Fallback for functions where signature inspection fails
            output.append(create_function_string(name, 'var_args', 'Any', docstring))
    return '\n'.join(output)


def create_stub(new_module_name: str, class_name: str, methods: str, import_statements: Sequence[str] = None) -> None:
    with open(f'_{new_module_name}_bridge_stub.py', mode='w', errors='ignore', encoding='ascii') as f:
        f.write(f'# Import required names here')
        if import_statements:
            f.write(f'\n{'\n'.join(import_statements)}')
        f.write(f'\n\n# noinspection PyUnresolvedReferences,HttpUrlsUsage,PyIncorrectDocstring,PyPep8Naming,PyShadowingBuiltins,PyDefaultArgument')
        f.write(f'\nclass {class_name}:\n')
        f.write(methods)
        f.write('\n')


if __name__ == "__main__":
    pass
    # ================ Create stub for the Symbolic Bridge ================
    # import _symbolic_bridge_module
    # create_stub('symbolic',
    #             'SympyModule',
    #             generate_method_signatures(_symbolic_bridge_module,
    #                                        truncate_docs_at=('\nExamples', '\n>>>')),
    #             import_statements=('from sympy import Expr',))

    pass
    # ================ Create stub for the Symbolic Bridge ================
    # import _repr_bridge_module
    # create_stub('repr',
    #             'SympyReprModule',
    #             generate_method_signatures(_repr_bridge_module,
    #                                        truncate_docs_at='\nExamples'),
    #             import_statements=None)

    pass
    # ================ Create stub for the Numerical Bridge ================
    # import _numerical_bridge_module
    # create_stub('numerical',
    #             'ScipyModule',
    #             generate_method_signatures(_numerical_bridge_module,
    #                                        truncate_docs_at=('\nReferences', '\nExamples')),
    #             import_statements=('import numpy as np',
    #                                'import numpy',
    #                                'from numpy import fmin',
    #                                'from math import inf',
    #                                'from typing import Union, Iterable, Optional, Callable',
    #                                'from scipy.optimize import Bounds, OptimizeResult'))
