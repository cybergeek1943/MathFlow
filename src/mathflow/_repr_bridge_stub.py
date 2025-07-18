# Import required names here

# noinspection PyUnresolvedReferences,HttpUrlsUsage,PyIncorrectDocstring,PyPep8Naming,PyShadowingBuiltins,PyDefaultArgument
class SympyReprModule:
    def ccode(self, assign_to=None, standard='c99', **settings):
        r"""Converts an expr to a string of c code
        
        Parameters
        ==========
        
        expr : Expr
            A SymPy expression to be converted.
        assign_to : optional
            When given, the argument is used as the name of the variable to which
            the expression is assigned. Can be a string, ``Symbol``,
            ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
            line-wrapping, or for expressions that generate multi-line statements.
        standard : str, optional
            String specifying the standard. If your compiler supports a more modern
            standard you may set this to 'c99' to allow the printer to use more math
            functions. [default='c89'].
        precision : integer, optional
            The precision for numbers such as pi [default=17].
        user_functions : dict, optional
            A dictionary where the keys are string representations of either
            ``FunctionClass`` or ``UndefinedFunction`` instances and the values
            are their desired C string representations. Alternatively, the
            dictionary value can be a list of tuples i.e. [(argument_test,
            cfunction_string)] or [(argument_test, cfunction_formater)]. See below
            for examples.
        dereference : iterable, optional
            An iterable of symbols that should be dereferenced in the printed code
            expression. These would be values passed by address to the function.
            For example, if ``dereference=[a]``, the resulting code would print
            ``(*a)`` instead of ``a``.
        human : bool, optional
            If True, the result is a single string that may contain some constant
            declarations for the number symbols. If False, the same information is
            returned in a tuple of (symbols_to_declare, not_supported_functions,
            code_text). [default=True].
        contract: bool, optional
            If True, ``Indexed`` instances are assumed to obey tensor contraction
            rules and the corresponding nested loops over indices are generated.
            Setting contract=False will not generate loops, instead the user is
            responsible to provide values for the indices in the code.
            [default=True].
        """
    def cxxcode(self, assign_to=None, standard='c++11', **settings):
        r"""C++ equivalent of :func:`~.ccode`. """
    def fcode(self, assign_to=None, **settings):
        r"""Converts an expr to a string of fortran code
        
        Parameters
        ==========
        
        expr : Expr
            A SymPy expression to be converted.
        assign_to : optional
            When given, the argument is used as the name of the variable to which
            the expression is assigned. Can be a string, ``Symbol``,
            ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
            line-wrapping, or for expressions that generate multi-line statements.
        precision : integer, optional
            DEPRECATED. Use type_mappings instead. The precision for numbers such
            as pi [default=17].
        user_functions : dict, optional
            A dictionary where keys are ``FunctionClass`` instances and values are
            their string representations. Alternatively, the dictionary value can
            be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
            for examples.
        human : bool, optional
            If True, the result is a single string that may contain some constant
            declarations for the number symbols. If False, the same information is
            returned in a tuple of (symbols_to_declare, not_supported_functions,
            code_text). [default=True].
        contract: bool, optional
            If True, ``Indexed`` instances are assumed to obey tensor contraction
            rules and the corresponding nested loops over indices are generated.
            Setting contract=False will not generate loops, instead the user is
            responsible to provide values for the indices in the code.
            [default=True].
        source_format : optional
            The source format can be either 'fixed' or 'free'. [default='fixed']
        standard : integer, optional
            The Fortran standard to be followed. This is specified as an integer.
            Acceptable standards are 66, 77, 90, 95, 2003, and 2008. Default is 77.
            Note that currently the only distinction internally is between
            standards before 95, and those 95 and after. This may change later as
            more features are added.
        name_mangling : bool, optional
            If True, then the variables that would become identical in
            case-insensitive Fortran are mangled by appending different number
            of ``_`` at the end. If False, SymPy Will not interfere with naming of
            variables. [default=True]
        """
    def jscode(self, assign_to=None, **settings):
        r"""Converts an expr to a string of javascript code
        
        Parameters
        ==========
        
        expr : Expr
            A SymPy expression to be converted.
        assign_to : optional
            When given, the argument is used as the name of the variable to which
            the expression is assigned. Can be a string, ``Symbol``,
            ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
            line-wrapping, or for expressions that generate multi-line statements.
        precision : integer, optional
            The precision for numbers such as pi [default=15].
        user_functions : dict, optional
            A dictionary where keys are ``FunctionClass`` instances and values are
            their string representations. Alternatively, the dictionary value can
            be a list of tuples i.e. [(argument_test, js_function_string)]. See
            below for examples.
        human : bool, optional
            If True, the result is a single string that may contain some constant
            declarations for the number symbols. If False, the same information is
            returned in a tuple of (symbols_to_declare, not_supported_functions,
            code_text). [default=True].
        contract: bool, optional
            If True, ``Indexed`` instances are assumed to obey tensor contraction
            rules and the corresponding nested loops over indices are generated.
            Setting contract=False will not generate loops, instead the user is
            responsible to provide values for the indices in the code.
            [default=True].
        """
    def julia_code(self, assign_to=None, **settings):
        r"""Converts `expr` to a string of Julia code.
        
        Parameters
        ==========
        
        expr : Expr
            A SymPy expression to be converted.
        assign_to : optional
            When given, the argument is used as the name of the variable to which
            the expression is assigned.  Can be a string, ``Symbol``,
            ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
            expressions that generate multi-line statements.
        precision : integer, optional
            The precision for numbers such as pi  [default=16].
        user_functions : dict, optional
            A dictionary where keys are ``FunctionClass`` instances and values are
            their string representations.  Alternatively, the dictionary value can
            be a list of tuples i.e. [(argument_test, cfunction_string)].  See
            below for examples.
        human : bool, optional
            If True, the result is a single string that may contain some constant
            declarations for the number symbols.  If False, the same information is
            returned in a tuple of (symbols_to_declare, not_supported_functions,
            code_text).  [default=True].
        contract: bool, optional
            If True, ``Indexed`` instances are assumed to obey tensor contraction
            rules and the corresponding nested loops over indices are generated.
            Setting contract=False will not generate loops, instead the user is
            responsible to provide values for the indices in the code.
            [default=True].
        inline: bool, optional
            If True, we try to create single-statement code instead of multiple
            statements.  [default=True].
        """
    def latex(self, full_prec=False, fold_frac_powers=False, fold_func_brackets=False, fold_short_frac=None, inv_trig_style='abbreviated', itex=False, ln_notation=False, long_frac_ratio=None, mat_delim='[', mat_str=None, mode='plain', mul_symbol=None, order=None, symbol_names={}, root_notation=True, mat_symbol_style='plain', imaginary_unit='i', gothic_re_im=False, decimal_separator='period', perm_cyclic=True, parenthesize_super=True, min=None, max=None, diff_operator='d', adjoint_style='dagger', disable_split_super_sub=False):
        r"""Convert the given expression to LaTeX string representation.
        
        Parameters
        ==========
        full_prec: boolean, optional
            If set to True, a floating point number is printed with full precision.
        fold_frac_powers : boolean, optional
            Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
        fold_func_brackets : boolean, optional
            Fold function brackets where applicable.
        fold_short_frac : boolean, optional
            Emit ``p / q`` instead of ``\frac{p}{q}`` when the denominator is
            simple enough (at most two terms and no powers). The default value is
            ``True`` for inline mode, ``False`` otherwise.
        inv_trig_style : string, optional
            How inverse trig functions should be displayed. Can be one of
            ``'abbreviated'``, ``'full'``, or ``'power'``. Defaults to
            ``'abbreviated'``.
        itex : boolean, optional
            Specifies if itex-specific syntax is used, including emitting
            ``$$...$$``.
        ln_notation : boolean, optional
            If set to ``True``, ``\ln`` is used instead of default ``\log``.
        long_frac_ratio : float or None, optional
            The allowed ratio of the width of the numerator to the width of the
            denominator before the printer breaks off long fractions. If ``None``
            (the default value), long fractions are not broken up.
        mat_delim : string, optional
            The delimiter to wrap around matrices. Can be one of ``'['``, ``'('``,
            or the empty string ``''``. Defaults to ``'['``.
        mat_str : string, optional
            Which matrix environment string to emit. ``'smallmatrix'``,
            ``'matrix'``, ``'array'``, etc. Defaults to ``'smallmatrix'`` for
            inline mode, ``'matrix'`` for matrices of no more than 10 columns, and
            ``'array'`` otherwise.
        mode: string, optional
            Specifies how the generated code will be delimited. ``mode`` can be one
            of ``'plain'``, ``'inline'``, ``'equation'`` or ``'equation*'``.  If
            ``mode`` is set to ``'plain'``, then the resulting code will not be
            delimited at all (this is the default). If ``mode`` is set to
            ``'inline'`` then inline LaTeX ``$...$`` will be used. If ``mode`` is
            set to ``'equation'`` or ``'equation*'``, the resulting code will be
            enclosed in the ``equation`` or ``equation*`` environment (remember to
            import ``amsmath`` for ``equation*``), unless the ``itex`` option is
            set. In the latter case, the ``$$...$$`` syntax is used.
        mul_symbol : string or None, optional
            The symbol to use for multiplication. Can be one of ``None``,
            ``'ldot'``, ``'dot'``, or ``'times'``.
        order: string, optional
            Any of the supported monomial orderings (currently ``'lex'``,
            ``'grlex'``, or ``'grevlex'``), ``'old'``, and ``'none'``. This
            parameter does nothing for `~.Mul` objects. Setting order to ``'old'``
            uses the compatibility ordering for ``~.Add`` defined in Printer. For
            very large expressions, set the ``order`` keyword to ``'none'`` if
            speed is a concern.
        symbol_names : dictionary of strings mapped to symbols, optional
            Dictionary of symbols and the custom strings they should be emitted as.
        root_notation : boolean, optional
            If set to ``False``, exponents of the form 1/n are printed in fractonal
            form. Default is ``True``, to print exponent in root form.
        mat_symbol_style : string, optional
            Can be either ``'plain'`` (default) or ``'bold'``. If set to
            ``'bold'``, a `~.MatrixSymbol` A will be printed as ``\mathbf{A}``,
            otherwise as ``A``.
        imaginary_unit : string, optional
            String to use for the imaginary unit. Defined options are ``'i'``
            (default) and ``'j'``. Adding ``r`` or ``t`` in front gives ``\mathrm``
            or ``\text``, so ``'ri'`` leads to ``\mathrm{i}`` which gives
            `\mathrm{i}`.
        gothic_re_im : boolean, optional
            If set to ``True``, `\Re` and `\Im` is used for ``re`` and ``im``, respectively.
            The default is ``False`` leading to `\operatorname{re}` and `\operatorname{im}`.
        decimal_separator : string, optional
            Specifies what separator to use to separate the whole and fractional parts of a
            floating point number as in `2.5` for the default, ``period`` or `2{,}5`
            when ``comma`` is specified. Lists, sets, and tuple are printed with semicolon
            separating the elements when ``comma`` is chosen. For example, [1; 2; 3] when
            ``comma`` is chosen and [1,2,3] for when ``period`` is chosen.
        parenthesize_super : boolean, optional
            If set to ``False``, superscripted expressions will not be parenthesized when
            powered. Default is ``True``, which parenthesizes the expression when powered.
        min: Integer or None, optional
            Sets the lower bound for the exponent to print floating point numbers in
            fixed-point format.
        max: Integer or None, optional
            Sets the upper bound for the exponent to print floating point numbers in
            fixed-point format.
        diff_operator: string, optional
            String to use for differential operator. Default is ``'d'``, to print in italic
            form. ``'rd'``, ``'td'`` are shortcuts for ``\mathrm{d}`` and ``\text{d}``.
        adjoint_style: string, optional
            String to use for the adjoint symbol. Defined options are ``'dagger'``
            (default),``'star'``, and ``'hermitian'``.
        
        Notes
        =====
        
        Not using a print statement for printing, results in double backslashes for
        latex commands since that's the way Python escapes backslashes in strings.
        
        >>> from sympy import latex, Rational
        >>> from sympy.abc import tau
        >>> latex((2*tau)**Rational(7,2))
        '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'
        >>> print(latex((2*tau)**Rational(7,2)))
        8 \sqrt{2} \tau^{\frac{7}{2}}
        """
    def maple_code(self, assign_to=None, **settings):
        r"""Converts ``expr`` to a string of Maple code.
        
        Parameters
        ==========
        
        expr : Expr
            A SymPy expression to be converted.
        assign_to : optional
            When given, the argument is used as the name of the variable to which
            the expression is assigned.  Can be a string, ``Symbol``,
            ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
            expressions that generate multi-line statements.
        precision : integer, optional
            The precision for numbers such as pi  [default=16].
        user_functions : dict, optional
            A dictionary where keys are ``FunctionClass`` instances and values are
            their string representations.  Alternatively, the dictionary value can
            be a list of tuples i.e. [(argument_test, cfunction_string)].  See
            below for examples.
        human : bool, optional
            If True, the result is a single string that may contain some constant
            declarations for the number symbols.  If False, the same information is
            returned in a tuple of (symbols_to_declare, not_supported_functions,
            code_text).  [default=True].
        contract: bool, optional
            If True, ``Indexed`` instances are assumed to obey tensor contraction
            rules and the corresponding nested loops over indices are generated.
            Setting contract=False will not generate loops, instead the user is
            responsible to provide values for the indices in the code.
            [default=True].
        inline: bool, optional
            If True, we try to create single-statement code instead of multiple
            statements.  [default=True].
        
        """
    def mathematica_code(self, **settings):
        r"""Converts an expr to a string of the Wolfram Mathematica code
        """
    def mathml(self, printer='content', order=None, encoding='utf-8', fold_frac_powers=False, fold_func_brackets=False, fold_short_frac=None, inv_trig_style='abbreviated', ln_notation=False, long_frac_ratio=None, mat_delim='[', mat_symbol_style='plain', mul_symbol=None, root_notation=True, symbol_names={}, mul_symbol_mathml_numbers='&#xB7;', disable_split_super_sub=False):
        r"""Returns the MathML representation of expr. If printer is presentation
        then prints Presentation MathML else prints content MathML.
        """
    def octave_code(self, assign_to=None, **settings):
        r"""Converts `expr` to a string of Octave (or Matlab) code.
        
        The string uses a subset of the Octave language for Matlab compatibility.
        
        Parameters
        ==========
        
        expr : Expr
            A SymPy expression to be converted.
        assign_to : optional
            When given, the argument is used as the name of the variable to which
            the expression is assigned.  Can be a string, ``Symbol``,
            ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
            expressions that generate multi-line statements.
        precision : integer, optional
            The precision for numbers such as pi  [default=16].
        user_functions : dict, optional
            A dictionary where keys are ``FunctionClass`` instances and values are
            their string representations.  Alternatively, the dictionary value can
            be a list of tuples i.e. [(argument_test, cfunction_string)].  See
            below for examples.
        human : bool, optional
            If True, the result is a single string that may contain some constant
            declarations for the number symbols.  If False, the same information is
            returned in a tuple of (symbols_to_declare, not_supported_functions,
            code_text).  [default=True].
        contract: bool, optional
            If True, ``Indexed`` instances are assumed to obey tensor contraction
            rules and the corresponding nested loops over indices are generated.
            Setting contract=False will not generate loops, instead the user is
            responsible to provide values for the indices in the code.
            [default=True].
        inline: bool, optional
            If True, we try to create single-statement code instead of multiple
            statements.  [default=True].
        """
    def pretty(self, order=None, full_prec='auto', use_unicode=None, wrap_line=True, num_columns=None, use_unicode_sqrt_char=True, root_notation=True, mat_symbol_style='plain', imaginary_unit='i', perm_cyclic=True):
        r"""Returns a string containing the prettified form of expr.
        
        For information on keyword arguments see pretty_print function.
        
        """
    def pycode(self, **settings):
        r"""Converts an expr to a string of Python code
        
        Parameters
        ==========
        
        expr : Expr
            A SymPy expression.
        fully_qualified_modules : bool
            Whether or not to write out full module names of functions
            (``math.sin`` vs. ``sin``). default: ``True``.
        standard : str or None, optional
            Only 'python3' (default) is supported.
            This parameter may be removed in the future.
        """
    def python(self, **settings):
        r"""Return Python interpretation of passed expression
        (can be passed to the exec() function without any modifications)"""
    def rcode(self, assign_to=None, **settings):
        r"""Converts an expr to a string of r code
        
        Parameters
        ==========
        
        expr : Expr
            A SymPy expression to be converted.
        assign_to : optional
            When given, the argument is used as the name of the variable to which
            the expression is assigned. Can be a string, ``Symbol``,
            ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
            line-wrapping, or for expressions that generate multi-line statements.
        precision : integer, optional
            The precision for numbers such as pi [default=15].
        user_functions : dict, optional
            A dictionary where the keys are string representations of either
            ``FunctionClass`` or ``UndefinedFunction`` instances and the values
            are their desired R string representations. Alternatively, the
            dictionary value can be a list of tuples i.e. [(argument_test,
            rfunction_string)] or [(argument_test, rfunction_formater)]. See below
            for examples.
        human : bool, optional
            If True, the result is a single string that may contain some constant
            declarations for the number symbols. If False, the same information is
            returned in a tuple of (symbols_to_declare, not_supported_functions,
            code_text). [default=True].
        contract: bool, optional
            If True, ``Indexed`` instances are assumed to obey tensor contraction
            rules and the corresponding nested loops over indices are generated.
            Setting contract=False will not generate loops, instead the user is
            responsible to provide values for the indices in the code.
            [default=True].
        """
    def rust_code(self, assign_to=None, **settings):
        r"""Converts an expr to a string of Rust code
        
        Parameters
        ==========
        
        expr : Expr
            A SymPy expression to be converted.
        assign_to : optional
            When given, the argument is used as the name of the variable to which
            the expression is assigned. Can be a string, ``Symbol``,
            ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
            line-wrapping, or for expressions that generate multi-line statements.
        precision : integer, optional
            The precision for numbers such as pi [default=15].
        user_functions : dict, optional
            A dictionary where the keys are string representations of either
            ``FunctionClass`` or ``UndefinedFunction`` instances and the values
            are their desired C string representations. Alternatively, the
            dictionary value can be a list of tuples i.e. [(argument_test,
            cfunction_string)].  See below for examples.
        dereference : iterable, optional
            An iterable of symbols that should be dereferenced in the printed code
            expression. These would be values passed by address to the function.
            For example, if ``dereference=[a]``, the resulting code would print
            ``(*a)`` instead of ``a``.
        human : bool, optional
            If True, the result is a single string that may contain some constant
            declarations for the number symbols. If False, the same information is
            returned in a tuple of (symbols_to_declare, not_supported_functions,
            code_text). [default=True].
        contract: bool, optional
            If True, ``Indexed`` instances are assumed to obey tensor contraction
            rules and the corresponding nested loops over indices are generated.
            Setting contract=False will not generate loops, instead the user is
            responsible to provide values for the indices in the code.
            [default=True].
        """
    def smtlib_code(self, auto_assert=True, auto_declare=True, precision=None, symbol_table=None, known_types=None, known_constants=None, known_functions=None, prefix_expressions=None, suffix_expressions=None, log_warn=None):
        r"""Converts ``expr`` to a string of smtlib code.
        
        Parameters
        ==========
        
        expr : Expr | List[Expr]
            A SymPy expression or system to be converted.
        auto_assert : bool, optional
            If false, do not modify expr and produce only the S-Expression equivalent of expr.
            If true, assume expr is a system and assert each boolean element.
        auto_declare : bool, optional
            If false, do not produce declarations for the symbols used in expr.
            If true, prepend all necessary declarations for variables used in expr based on symbol_table.
        precision : integer, optional
            The ``evalf(..)`` precision for numbers such as pi.
        symbol_table : dict, optional
            A dictionary where keys are ``Symbol`` or ``Function`` instances and values are their Python type i.e. ``bool``, ``int``, ``float``, or ``Callable[...]``.
            If incomplete, an attempt will be made to infer types from ``expr``.
        known_types: dict, optional
            A dictionary where keys are ``bool``, ``int``, ``float`` etc. and values are their corresponding SMT type names.
            If not given, a partial listing compatible with several solvers will be used.
        known_functions : dict, optional
            A dictionary where keys are ``Function``, ``Relational``, ``BooleanFunction``, or ``Expr`` instances and values are their SMT string representations.
            If not given, a partial listing optimized for dReal solver (but compatible with others) will be used.
        known_constants: dict, optional
            A dictionary where keys are ``NumberSymbol`` instances and values are their SMT variable names.
            When using this feature, extra caution must be taken to avoid naming collisions between user symbols and listed constants.
            If not given, constants will be expanded inline i.e. ``3.14159`` instead of ``MY_SMT_VARIABLE_FOR_PI``.
        prefix_expressions: list, optional
            A list of lists of ``str`` and/or expressions to convert into SMTLib and prefix to the output.
        suffix_expressions: list, optional
            A list of lists of ``str`` and/or expressions to convert into SMTLib and postfix to the output.
        log_warn: lambda function, optional
            A function to record all warnings during potentially risky operations.
            Soundness is a core value in SMT solving, so it is good to log all assumptions made.
        """
    def tree(self, assumptions=True):
        r"""
        Returns a tree representation of "node" as a string.
        
        It uses print_node() together with pprint_nodes() on node.args recursively.
        
        Parameters
        ==========
        
        assumptions : bool, optional
            The flag to decide whether to print out all the assumption data
            (such as ``is_integer`, ``is_real``) associated with the
            expression or not.
        
            Enabling the flag makes the result verbose, and the printed
            result may not be deterministic because of the randomness used
            in backtracing the assumptions.
        
        See Also
        ========
        
        print_tree
        
        """
