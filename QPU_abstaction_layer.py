"""
optical_eval.py
================

An "eval-for-code" front end for optical_binary.py.

The original module only exposes two primitives:

    binary_multiply(x, y)   -> one hole pair + camera read (AND)
    binary_add_many(values) -> one plane of holes + camera read (SUM)

Everything else (loops, De Morgan expansions, arithmetic, control flow,
etc.) had to be hand-written against those two calls. This module gives
you three ways to *program* against the optical backend instead of
hand-wiring it:

1. Bit    - a wrapper around a nonnegative integer that overloads
            & | ^ ~  (boolean, bits only), + - *  (arithmetic), and
            == != < <= > >=  (comparison) so you can write ordinary
            Python expressions and have every operation actually
            execute through camera_exposure().

2. ooeval(expr, **vars) - a real "eval()" for a *single expression*
            string, same operator set as Bit.

3. oorun(code, **vars)  - a real "exec()" for a small *program*: plain
            assignment, if/elif/else, while, for name in range(...)/
            [list], break/continue, and print(...). Every expression
            evaluated while the program runs (loop bounds, conditions,
            assigned values) still goes through the same optical
            primitives, all logged into one continuous trace.

All three layers are built only from binary_multiply / binary_add_many /
apply_holes, so nothing here bypasses the "optics" - it's just a
friendlier way to drive it. Arithmetic (+, -, *) reuses the exact same
camera: an integer is represented as that many open holes (unary), so
"addition" is literally combining two hole-counts on one aperture and
reading the total intensity, "multiplication" is repeated addition, and
comparisons (==, <, etc.) reuse the subtractor - whichever direction of
a-b or b-a doesn't go negative tells you the ordering.

    +   optical_add(a, b)       open a+b holes, read the total count
    -   optical_subtract(a, b)  start with a holes open, close b of them,
                                 read what's left (uses apply_holes)
    *   optical_multiply(a, b)  b passes of optical_add, i.e. repeated
                                 optical addition (or a single AND
                                 exposure in the 0/1 x 0/1 case)
    ==, <, etc.  opt_cmp(a, b)  try a-b then b-a on the subtractor;
                                 whichever succeeds (and by how much)
                                 gives the ordering
"""

import ast

import numpy as np

from optical_binary import (
    binary_multiply,
    binary_add_many,
    make_binary_plane_and_holes,
    apply_holes,
    camera_exposure,
)


# =====================================================================
# THE OPTICAL BOOLEAN ALGEBRA
# =====================================================================
#
# The hardware only really gives us two operations: multiply (one
# hole-pair exposure) and sum (one plane-of-holes exposure). Every
# boolean operator below is defined in terms of just those two, so each
# call below is a real optical exposure, not a shortcut back to plain
# Python arithmetic.

def opt_and(x: int, y: int) -> int:
    """AND is exactly what the hardware does: multiply through one hole."""
    return binary_multiply(x, y)


def opt_or(x: int, y: int) -> int:
    """OR = sum the two bits on the camera, then clamp the exposure to {0,1}."""
    total = binary_add_many([x, y])
    return 1 if total >= 1 else 0


def opt_xor(x: int, y: int) -> int:
    """XOR = sum the two bits on the camera, keep it mod 2."""
    total = binary_add_many([x, y])
    return total % 2


def opt_not(x: int) -> int:
    """NOT = complement. (No exposure needed - it's just which hole is open.)"""
    if x not in (0, 1):
        raise ValueError("NOT expects a 0/1 value")
    return 1 - x


# =====================================================================
# THE OPTICAL ARITHMETIC
# =====================================================================
#
# The same aperture that sums bits for OR/XOR can sum *counts* of bits.
# Represent a nonnegative integer n as n open holes (a unary code). Then:
#   - addition   = put both operands' holes on one aperture, read the total
#   - subtraction = start with a's holes open, close b of them, read what's left
#   - multiplication = repeated addition (or a single AND exposure for bits)

def _require_nonneg_int(name, value):
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} expects a nonnegative integer, got {value!r}")
    return value


def optical_add(a: int, b: int) -> int:
    """a + b: open a+b holes on one aperture, read the total transmitted count."""
    _require_nonneg_int("optical_add", a)
    _require_nonneg_int("optical_add", b)
    values = [1] * a + [1] * b
    if not values:
        return 0
    return binary_add_many(values)


def optical_subtract(a: int, b: int) -> int:
    """a - b: start with a holes open, close b of them, read what's left."""
    _require_nonneg_int("optical_subtract", a)
    _require_nonneg_int("optical_subtract", b)
    if b > a:
        raise ValueError(
            "optical_subtract: an aperture can't hold a negative photon "
            f"count ({a} - {b} < 0)"
        )
    if a == 0:
        return 0

    plane, holes = make_binary_plane_and_holes(
        np.ones(a, dtype=np.uint8),
    )

    # Close b of the currently-open holes.
    flat_holes = holes.flatten()
    closed = 0
    for i in range(flat_holes.size):
        if flat_holes[i] == 1 and closed < b:
            flat_holes[i] = 0
            closed += 1
    holes = flat_holes.reshape(holes.shape)

    optics = camera_exposure(plane, holes)
    return int(round(optics["dc_field"].real))


def optical_multiply(a: int, b: int) -> int:
    """a * b: repeated optical addition (b passes), or one AND exposure for bits."""
    _require_nonneg_int("optical_multiply", a)
    _require_nonneg_int("optical_multiply", b)

    if a in (0, 1) and b in (0, 1):
        # Single-bit case: this *is* the hardware's native AND exposure.
        return binary_multiply(a, b)

    total = 0
    for _ in range(b):
        total = optical_add(total, a)
    return total


# =====================================================================
# THE OPTICAL COMPARISONS
# =====================================================================
#
# No new hardware needed here either: the subtractor already refuses to
# go negative. Try a-b; if that succeeds you know a >= b (and the result
# tells you if they're equal). If it fails, b must be strictly greater,
# so b-a is guaranteed to succeed instead. Two subtractor exposures,
# at most, tell you the full ordering.

def opt_cmp(a: int, b: int) -> int:
    """Return -1 / 0 / 1 for a<b / a==b / a>b using only the subtractor."""
    _require_nonneg_int("opt_cmp", a)
    _require_nonneg_int("opt_cmp", b)

    try:
        forward = optical_subtract(a, b)  # a - b; only succeeds if a >= b
    except ValueError:
        forward = None

    if forward is not None:
        return 0 if forward == 0 else 1

    backward = optical_subtract(b, a)  # a < b, so b - a is guaranteed to work
    return -1 if backward > 0 else 0


def opt_eq(a: int, b: int) -> int:
    return 1 if opt_cmp(a, b) == 0 else 0


def opt_ne(a: int, b: int) -> int:
    return 1 if opt_cmp(a, b) != 0 else 0


def opt_lt(a: int, b: int) -> int:
    return 1 if opt_cmp(a, b) < 0 else 0


def opt_le(a: int, b: int) -> int:
    return 1 if opt_cmp(a, b) <= 0 else 0


def opt_gt(a: int, b: int) -> int:
    return 1 if opt_cmp(a, b) > 0 else 0


def opt_ge(a: int, b: int) -> int:
    return 1 if opt_cmp(a, b) >= 0 else 0


# =====================================================================
# 1) Bit - operator-overloaded value, so plain Python syntax "is" the program
# =====================================================================

class Bit:
    """
    A nonnegative-integer value whose operators are wired straight into
    the optical primitives. Write normal Python:

        a, b, c = Bit(1), Bit(0), Bit(1)
        logic  = (a & b) | c         # & | ^ ~   -> boolean, bits only
        total  = a + b + c           # + - *     -> arithmetic, any nonneg int
        scaled = total * Bit(3)

    Boolean operators (& | ^ ~) still require 0/1 operands, since that's
    what the AND/OR/XOR exposures are defined over. Arithmetic operators
    (+ - *) accept any nonnegative integer, since they're built on the
    same aperture used in unary (open-hole-count) form.

    `result.trace` accumulates a log of every optical exposure that ran,
    in order, so you can see exactly which hole-pairs/planes were used.
    """

    def __init__(self, value, trace=None):
        value = int(value)
        if value < 0:
            raise ValueError("Bit must be a nonnegative integer")
        self.value = value
        self.trace = trace if trace is not None else []

    def _merge_trace(self, other):
        merged = list(self.trace)
        if isinstance(other, Bit):
            merged.extend(other.trace)
        return merged

    def _combine(self, other, op_name, op_fn):
        other_val = other.value if isinstance(other, Bit) else int(other)
        result = op_fn(self.value, other_val)
        trace = self._merge_trace(other)
        trace.append((op_name, (self.value, other_val), result))
        return Bit(result, trace=trace)

    # --- boolean (bits only) ---

    def __and__(self, other):
        return self._combine(other, "AND", opt_and)

    def __or__(self, other):
        return self._combine(other, "OR", opt_or)

    def __xor__(self, other):
        return self._combine(other, "XOR", opt_xor)

    def __invert__(self):
        result = opt_not(self.value)
        trace = list(self.trace)
        trace.append(("NOT", (self.value,), result))
        return Bit(result, trace=trace)

    # --- arithmetic (any nonnegative integer) ---

    def __add__(self, other):
        return self._combine(other, "ADD", optical_add)

    def __radd__(self, other):
        return self._combine(other, "ADD", lambda x, y: optical_add(y, x))

    def __sub__(self, other):
        return self._combine(other, "SUB", optical_subtract)

    def __rsub__(self, other):
        return self._combine(other, "SUB", lambda x, y: optical_subtract(y, x))

    def __mul__(self, other):
        return self._combine(other, "MUL", optical_multiply)

    def __rmul__(self, other):
        return self._combine(other, "MUL", lambda x, y: optical_multiply(y, x))

    # --- plumbing ---

    def __int__(self):
        return self.value

    def __bool__(self):
        return bool(self.value)

    def __eq__(self, other):
        other_val = other.value if isinstance(other, Bit) else other
        return opt_eq(self.value, int(other_val)) == 1

    def __ne__(self, other):
        other_val = other.value if isinstance(other, Bit) else other
        return opt_ne(self.value, int(other_val)) == 1

    def __lt__(self, other):
        other_val = other.value if isinstance(other, Bit) else other
        return opt_lt(self.value, int(other_val)) == 1

    def __le__(self, other):
        other_val = other.value if isinstance(other, Bit) else other
        return opt_le(self.value, int(other_val)) == 1

    def __gt__(self, other):
        other_val = other.value if isinstance(other, Bit) else other
        return opt_gt(self.value, int(other_val)) == 1

    def __ge__(self, other):
        other_val = other.value if isinstance(other, Bit) else other
        return opt_ge(self.value, int(other_val)) == 1

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"Bit({self.value})"

    def print_trace(self):
        for step, (op, inputs, result) in enumerate(self.trace, start=1):
            print(f"  [{step}] {op}{inputs} -> {result}")


# =====================================================================
# 2) ooeval - a real eval() for text expressions
# =====================================================================

# Which Python AST operators map to which optical primitive.
_BINOP_TABLE = {
    ast.BitAnd: opt_and,
    ast.BitOr: opt_or,
    ast.BitXor: opt_xor,
    ast.Add: optical_add,
    ast.Sub: optical_subtract,
    ast.Mult: optical_multiply,
}

_BOOLOP_TABLE = {
    ast.And: opt_and,
    ast.Or: opt_or,
}

_COMPARE_TABLE = {
    ast.Eq: opt_eq,
    ast.NotEq: opt_ne,
    ast.Lt: opt_lt,
    ast.LtE: opt_le,
    ast.Gt: opt_gt,
    ast.GtE: opt_ge,
}


class _OpticalExpressionEvaluator(ast.NodeVisitor):
    """Walks a parsed expression tree, firing one optical exposure per node."""

    def __init__(self, variables, trace=None):
        self.variables = variables
        # `trace` can be a list shared with a caller (e.g. oorun, running a
        # whole program) so every exposure across many expressions lands
        # in one continuous log; if omitted, this evaluator keeps its own.
        self.trace = trace if trace is not None else []

    def _record(self, op_name, inputs, result):
        self.trace.append((op_name, inputs, result))
        return result

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Name(self, node):
        if node.id not in self.variables:
            raise NameError(f"unknown variable '{node.id}' in expression")
        value = int(self.variables[node.id])
        if value < 0:
            raise ValueError(f"variable '{node.id}' must be a nonnegative integer")
        return value

    def visit_Constant(self, node):
        if not isinstance(node.value, int) or node.value < 0:
            raise ValueError("literals must be nonnegative integers")
        return int(node.value)

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in _BINOP_TABLE:
            raise SyntaxError(
                f"unsupported operator {op_type.__name__}; use & | ^ ~ + - *"
            )
        left = self.visit(node.left)
        right = self.visit(node.right)
        result = _BINOP_TABLE[op_type](left, right)
        return self._record(op_type.__name__, (left, right), result)

    def visit_BoolOp(self, node):
        # supports Python's "and" / "or" keywords too, folded left to right
        op_type = type(node.op)
        fn = _BOOLOP_TABLE[op_type]
        values = [self.visit(v) for v in node.values]
        acc = values[0]
        for v in values[1:]:
            new_acc = fn(acc, v)
            self._record(op_type.__name__, (acc, v), new_acc)
            acc = new_acc
        return acc

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, (ast.Invert, ast.Not)):
            raise SyntaxError("unsupported unary operator; use ~ or 'not'")
        operand = self.visit(node.operand)
        result = opt_not(operand)
        return self._record("NOT", (operand,), result)

    def visit_Compare(self, node):
        # a < b < c chains like Python does: each link is its own optical
        # comparison, and the links are combined with a real AND exposure.
        left = self.visit(node.left)
        combined = 1
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _COMPARE_TABLE:
                raise SyntaxError(
                    f"unsupported comparison {op_type.__name__}; "
                    "use == != < <= > >="
                )
            right = self.visit(comparator)
            link = _COMPARE_TABLE[op_type](left, right)
            self._record(op_type.__name__, (left, right), link)
            combined = self._record("AND", (combined, link), opt_and(combined, link))
            left = right
        return combined

    def generic_visit(self, node):
        raise SyntaxError(f"unsupported syntax: {type(node).__name__}")


def ooeval(expr: str, return_trace: bool = False, **variables):
    """
    Evaluate a boolean/bitwise expression string through the optical backend.

    Example:
        ooeval("(a & b) | ~c", a=1, b=0, c=1)   -> 0
        ooeval("a and b or c", a=1, b=1, c=0, return_trace=True)
        ooeval("a + b * c", a=2, b=3, c=4)      -> 14
        ooeval("(a + b) - c", a=5, b=2, c=3)    -> 4
        ooeval("a == b", a=3, b=3)              -> 1
        ooeval("a < b < c", a=1, b=2, c=3)      -> 1

    Supported syntax: & | ^ ~  (boolean, 0/1 operands only),
    + - *  (arithmetic, any nonnegative integer), and
    == != < <= > >=  (comparisons, any nonnegative integer), plus the
    keywords and/or/not, parentheses, and variables/literals - ordinary
    Python boolean/arithmetic/comparison syntax, minus negative numbers
    (apertures can't hold a negative photon count).
    """

    tree = ast.parse(expr, mode="eval")
    evaluator = _OpticalExpressionEvaluator(variables)
    result = evaluator.visit(tree)

    if return_trace:
        return result, evaluator.trace
    return result


# =====================================================================
# 3) oorun - a small programming language: assignment, if/elif/else,
#    while, for, break/continue, print - all sitting on the same optics
# =====================================================================

class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


_MAX_LOOP_ITERATIONS = 100_000  # safety valve against runaway while-loops


class _OpticalStatementExecutor(ast.NodeVisitor):
    """
    Walks a parsed *program* (ast.Module) statement by statement. Every
    expression it evaluates - conditions, loop bounds, assigned values -
    goes through _OpticalExpressionEvaluator, so every &, |, ^, ~, +, -,
    *, ==, <, and so on inside the program is a real optical exposure,
    logged into one continuous, shared trace.
    """

    def __init__(self, variables, trace, output):
        self.variables = variables
        self.trace = trace
        self.output = output

    def _eval(self, node):
        evaluator = _OpticalExpressionEvaluator(self.variables, self.trace)
        return evaluator.visit(node)

    def _run_block(self, statements):
        for stmt in statements:
            self.visit(stmt)

    # --- module / simple statements ---

    def visit_Module(self, node):
        self._run_block(node.body)

    def visit_Pass(self, node):
        pass

    def visit_Break(self, node):
        raise _BreakSignal()

    def visit_Continue(self, node):
        raise _ContinueSignal()

    def visit_Assign(self, node):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise SyntaxError("only simple 'name = expr' assignment is supported")
        self.variables[node.targets[0].id] = self._eval(node.value)

    def visit_AugAssign(self, node):
        if not isinstance(node.target, ast.Name):
            raise SyntaxError("only 'name += expr' style augmented assignment is supported")
        name = node.target.id
        if name not in self.variables:
            raise NameError(f"unknown variable '{name}'")
        op_type = type(node.op)
        if op_type not in _BINOP_TABLE:
            raise SyntaxError(f"unsupported augmented operator {op_type.__name__}")
        current = self.variables[name]
        rhs = self._eval(node.value)
        result = _BINOP_TABLE[op_type](current, rhs)
        self.trace.append((op_type.__name__, (current, rhs), result))
        self.variables[name] = result

    def visit_Expr(self, node):
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "print"
        ):
            values = [self._eval(arg) for arg in call.args]
            line = " ".join(str(v) for v in values)
            self.output.append(line)
            print(line)
            return
        # any other bare expression is still evaluated (and still fires
        # its optics), just discarded - mirrors Python's own behaviour
        self._eval(call)

    # --- control flow ---

    def visit_If(self, node):
        branch = node.body if self._eval(node.test) else node.orelse
        self._run_block(branch)

    def visit_While(self, node):
        iterations = 0
        while self._eval(node.test):
            iterations += 1
            if iterations > _MAX_LOOP_ITERATIONS:
                raise RuntimeError(
                    f"while loop exceeded {_MAX_LOOP_ITERATIONS} iterations"
                )
            try:
                self._run_block(node.body)
            except _BreakSignal:
                break
            except _ContinueSignal:
                continue

    def visit_For(self, node):
        if not isinstance(node.target, ast.Name):
            raise SyntaxError("only 'for name in ...' loops are supported")

        for value in self._resolve_iterable(node.iter):
            self.variables[node.target.id] = value
            try:
                self._run_block(node.body)
            except _BreakSignal:
                break
            except _ContinueSignal:
                continue

    def _resolve_iterable(self, node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
        ):
            bounds = [self._eval(arg) for arg in node.args]
            return range(*bounds)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(elt) for elt in node.elts]
        raise SyntaxError(
            "for-loops only support range(...) or a literal [list]"
        )

    def generic_visit(self, node):
        raise SyntaxError(f"unsupported statement: {type(node).__name__}")


def oorun(code: str, return_trace: bool = False, **initial_variables):
    """
    Run a small program through the optical backend: assignment,
    if/elif/else, while, for, break/continue, and print(). Every
    arithmetic/boolean/comparison operation the program performs is a
    real exposure through camera_exposure(), logged in order.

    Example:
        oorun('''
        total = 0
        for i in range(5):
            total = total + i
        print(total)
        ''')  # prints 10, returns {'total': 10, 'i': 4}

        oorun('''
        if a > b:
            bigger = a
        else:
            bigger = b
        ''', a=3, b=7)  # -> {'a': 3, 'b': 7, 'bigger': 7}

    Supported syntax: = += (with & | ^ ^ + - * as the op), if/elif/else,
    while, for name in range(...) / [literal list], break, continue,
    print(...), and any expression ooeval() accepts. Nonnegative
    integers only - no negative numbers, no floats, no lists as values.

    Returns the final variables dict; pass return_trace=True to also get
    (trace, output_lines).
    """

    tree = ast.parse(code, mode="exec")
    variables = dict(initial_variables)
    trace = []
    output = []

    executor = _OpticalStatementExecutor(variables, trace, output)
    executor.visit(tree)

    if return_trace:
        return variables, trace, output
    return variables


# =====================================================================
# DEMO: the old hand-written triple loop, now just... normal code
# =====================================================================

def demo_bit_style(A, B, C):
    """
    This reproduces evaluate_3d_binary_loop from optical_binary.py, but
    instead of manually calling binary_multiply/binary_add_many, it's
    just plain Python with Bit values - each &, |, ~ below is a real
    optical exposure under the hood.
    """
    terms = []
    for a in A:
        for b in B:
            for c in C:
                term = Bit(a) & Bit(b) & Bit(c)
                terms.append(term.value)
    total = binary_add_many(terms)
    return total


def demo_ooeval_style():
    """A handful of expressions run straight through ooeval()."""
    examples = [
        ("a & b", dict(a=1, b=1)),
        ("a & b & c", dict(a=1, b=1, c=0)),
        ("a | b", dict(a=0, b=1)),
        ("a ^ b", dict(a=1, b=1)),
        ("~a & b", dict(a=0, b=1)),
        ("(a & b) | (~c & d)", dict(a=1, b=0, c=1, d=1)),
        ("a and b or c", dict(a=0, b=1, c=0)),
        ("a + b", dict(a=2, b=3)),
        ("a + b * c", dict(a=2, b=3, c=4)),
        ("(a + b) - c", dict(a=5, b=2, c=3)),
        ("a * b * c", dict(a=2, b=3, c=4)),
        ("a == b", dict(a=3, b=3)),
        ("a < b < c", dict(a=1, b=2, c=3)),
    ]

    for expr, values in examples:
        result, trace = ooeval(expr, return_trace=True, **values)
        print(f"ooeval({expr!r}, {values}) = {result}")
        for step, (op, inputs, out) in enumerate(trace, start=1):
            print(f"    [{step}] {op}{inputs} -> {out}")


def demo_oorun_style():
    """A couple of small programs run through oorun()."""

    print("-- sum 0..4 with a for-loop --")
    program = """
total = 0
for i in range(5):
    total = total + i
print(total)
"""
    variables, trace, _ = oorun(program, return_trace=True)
    print(f"final variables: {variables}")
    print(f"optical exposures fired: {len(trace)}")

    print()
    print("-- if/elif/else + comparisons + a while loop --")
    program = """
n = 1
while n <= limit:
    if n == target:
        print(n, 999)
    elif n < target:
        print(n, 1)
    else:
        print(n, 2)
    n = n + 1
"""
    variables, trace, output = oorun(
        program, return_trace=True, limit=5, target=3,
    )
    print(f"final variables: {variables}")
    print(f"optical exposures fired: {len(trace)}")

    print()
    print("-- for-loop + break: find the first n where n*n == target --")
    program = """
found = 0
for n in range(1, limit):
    if n * n == target:
        found = n
        break
print(found)
"""
    variables, trace, output = oorun(
        program, return_trace=True, limit=20, target=49,
    )
    print(f"final variables: {variables}")
    print(f"optical exposures fired: {len(trace)}")


def interactive_program():
    """
    Read a multi-line oorun() program from stdin (blank line ends input),
    then run it. Optionally seed starting variables with 'name=value'
    pairs first.
    """
    print("Enter initial values (e.g. limit=10), blank to skip:")
    values_str = input("> ").strip()
    values = {}
    for pair in values_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, val = pair.split("=", 1)
        values[name.strip()] = int(val.strip())

    print("Enter your program, blank line to run it:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    code = "\n".join(lines)

    variables, trace, output = oorun(code, return_trace=True, **values)
    print(f"\nfinal variables: {variables}")
    print(f"optical exposures fired: {len(trace)}")
    return variables


def interactive_eval():
    """
    Prompt for an expression and its variable values, then run it
    through ooeval(). Values are entered as "a=1,b=0,c=1" and can be any
    nonnegative integer for arithmetic (+ - *); boolean operators
    (& | ^ ~) still require 0/1.

    Example session:
        Enter expression: (a & b) | ~c
        Enter values: a=1,b=0,c=1
        ooeval('(a & b) | ~c', {'a': 1, 'b': 0, 'c': 1}) = 0
            [1] BitAnd(1, 0) -> 0
            [2] NOT(1,) -> 0
            [3] BitOr(0, 0) -> 0

        Enter expression: a + b * c
        Enter values: a=2,b=3,c=4
        ooeval('a + b * c', {'a': 2, 'b': 3, 'c': 4}) = 14
    """

    expr = input("Enter expression: ").strip()
    values_str = input("Enter values (e.g. a=2,b=3,c=4): ").strip()

    values = {}
    for pair in values_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"expected 'name=value', got '{pair}'"
            )
        name, val = pair.split("=", 1)
        name = name.strip()
        val = int(val.strip())
        if val < 0:
            raise ValueError(f"'{name}' must be a nonnegative integer, got {val}")
        values[name] = val

    result, trace = ooeval(expr, return_trace=True, **values)

    print(f"\nooeval({expr!r}, {values}) = {result}")
    for step, (op, inputs, out) in enumerate(trace, start=1):
        print(f"    [{step}] {op}{inputs} -> {out}")

    return result


def run_program_file(path: str, **initial_variables):
    """
    Load and run an oorun() program from a file, e.g.:
        python3 optical_eval.py --file code.q target=16 limit=20
    """
    with open(path, "r") as f:
        code = f.read()

    variables, trace, output = oorun(code, return_trace=True, **initial_variables)

    print(f"\nfinal variables: {variables}")
    print(f"optical exposures fired: {len(trace)}")
    return variables


if __name__ == "__main__":
    import sys

    if "--interactive" in sys.argv or "-i" in sys.argv:
        interactive_eval()
        raise SystemExit(0)

    if "--program" in sys.argv or "-p" in sys.argv:
        interactive_program()
        raise SystemExit(0)

    if "--file" in sys.argv or "-f" in sys.argv:
        flag = "--file" if "--file" in sys.argv else "-f"
        idx = sys.argv.index(flag)
        file_path = sys.argv[idx + 1]

        file_values = {}
        for kv in sys.argv[idx + 2:]:
            if "=" in kv:
                name, val = kv.split("=", 1)
                file_values[name.strip()] = int(val.strip())

        run_program_file(file_path, **file_values)
        raise SystemExit(0)

    
