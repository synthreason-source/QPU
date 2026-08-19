"""
optical_eval.py
================

An "eval-for-code" front end for optical_binary.py.

The original module only exposes two primitives:

    binary_multiply(x, y)   -> one hole pair + camera read (AND)
    binary_add_many(values) -> one plane of holes + camera read (SUM)

Everything else (loops, De Morgan expansions, arithmetic, etc.) had to be
hand-written against those two calls. This module gives you two ways to
*program* against the optical backend instead of hand-wiring it:

1. Bit  - a wrapper around a nonnegative integer that overloads
          & | ^ ~  (boolean, bits only) and + - *  (arithmetic, any
          nonnegative integer) so you can write ordinary Python
          expressions and have every operation actually execute through
          camera_exposure().

2. qeval(expr, **vars) - a real "eval()" that takes a *string* of
          Python-flavoured boolean/bitwise/arithmetic syntax and runs it
          through the same optical primitives, returning the result
          (and, optionally, a step-by-step trace of every optical
          exposure that fired).

Both layers are built only from binary_multiply / binary_add_many /
apply_holes, so nothing here bypasses the "optics" - it's just a
friendlier way to drive it. Arithmetic (+, -, *) reuses the exact same
camera: an integer is represented as that many open holes (unary), so
"addition" is literally combining two hole-counts on one aperture and
reading the total intensity, and "multiplication" is repeated addition.

    +   optical_add(a, b)       open a+b holes, read the total count
    -   optical_subtract(a, b)  start with a holes open, close b of them,
                                 read what's left (uses apply_holes)
    *   optical_multiply(a, b)  b passes of optical_add, i.e. repeated
                                 optical addition (or a single AND
                                 exposure in the 0/1 x 0/1 case)
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
        return self.value == other_val

    def __repr__(self):
        return f"Bit({self.value})"

    def print_trace(self):
        for step, (op, inputs, result) in enumerate(self.trace, start=1):
            print(f"  [{step}] {op}{inputs} -> {result}")


# =====================================================================
# 2) qeval - a real eval() for text expressions
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


class _OpticalExpressionEvaluator(ast.NodeVisitor):
    """Walks a parsed expression tree, firing one optical exposure per node."""

    def __init__(self, variables):
        self.variables = variables
        self.trace = []

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

    def generic_visit(self, node):
        raise SyntaxError(f"unsupported syntax: {type(node).__name__}")


def qeval(expr: str, return_trace: bool = False, **variables):
    """
    Evaluate a boolean/bitwise expression string through the optical backend.

    Example:
        qeval("(a & b) | ~c", a=1, b=0, c=1)   -> 0
        qeval("a and b or c", a=1, b=1, c=0, return_trace=True)
        qeval("a + b * c", a=2, b=3, c=4)      -> 14
        qeval("(a + b) - c", a=5, b=2, c=3)    -> 4

    Supported syntax: & | ^ ~  (boolean, 0/1 operands only) and
    + - *  (arithmetic, any nonnegative integer), plus the keywords
    and/or/not, parentheses, and variables/literals - ordinary Python
    boolean + arithmetic syntax, minus negative numbers (apertures can't
    hold a negative photon count).
    """

    tree = ast.parse(expr, mode="eval")
    evaluator = _OpticalExpressionEvaluator(variables)
    result = evaluator.visit(tree)

    if return_trace:
        return result, evaluator.trace
    return result


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

def interactive_eval():
    """
    Prompt for an expression and its variable values, then run it
    through qeval(). Values are entered as "a=1,b=0,c=1" and can be any
    nonnegative integer for arithmetic (+ - *); boolean operators
    (& | ^ ~) still require 0/1.

    Example session:
        Enter expression: (a & b) | ~c
        Enter values: a=1,b=0,c=1
        qeval('(a & b) | ~c', {'a': 1, 'b': 0, 'c': 1}) = 0
            [1] BitAnd(1, 0) -> 0
            [2] NOT(1,) -> 0
            [3] BitOr(0, 0) -> 0

        Enter expression: a + b * c
        Enter values: a=2,b=3,c=4
        qeval('a + b * c', {'a': 2, 'b': 3, 'c': 4}) = 14
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

    result, trace = qeval(expr, return_trace=True, **values)

    print(f"\nqeval({expr!r}, {values}) = {result}")
    for step, (op, inputs, out) in enumerate(trace, start=1):
        print(f"    [{step}] {op}{inputs} -> {out}")

    return result


if __name__ == "__main__":
    import sys
    
    print("""examples = [
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
    ]""")
    interactive_eval()
