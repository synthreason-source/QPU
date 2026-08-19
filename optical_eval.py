import math
import random
import time

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


np.set_printoptions(
    precision=3,
    suppress=True,
)


# =====================================================================
# BINARY 2D PLANE, HOLES, AND CAMERA
# =====================================================================

def fft2c(
    field: np.ndarray,
) -> np.ndarray:
    """Return a centered 2D Fourier transform."""

    field = np.asarray(
        field,
        dtype=complex,
    )

    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(field),
        ),
    )


def to_square_plane(
    values: np.ndarray,
) -> np.ndarray:
    """Place values on a square binary 2D plane."""

    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional"
        )

    if not np.all(
        np.isfinite(values),
    ):
        raise ValueError(
            "values must be finite"
        )

    if not np.all(
        values == values.astype(int),
    ):
        raise ValueError(
            "values must contain integers"
        )

    if np.any(values < 0):
        raise ValueError(
            "values must be nonnegative"
        )

    count = values.size

    side = int(
        np.ceil(
            np.sqrt(
                max(count, 1),
            ),
        ),
    )

    plane = np.zeros(
        (side, side),
        dtype=np.uint8,
    )

    plane.flat[:count] = values.astype(
        np.uint8,
    )

    if not np.all(
        np.isin(
            plane,
            [0, 1],
        ),
    ):
        raise ValueError(
            "the 2D plane must be binary"
        )

    return plane


def make_binary_plane_and_holes(
    values: np.ndarray,
):
    """Create a binary 2D plane and matching holes."""

    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional"
        )

    if not np.all(
        np.isin(
            values,
            [0, 1],
        ),
    ):
        raise ValueError(
            "values must contain only 0 or 1"
        )

    plane = to_square_plane(
        values,
    )

    holes = np.zeros(
        plane.shape,
        dtype=np.uint8,
    )

    holes.flat[:values.size] = 1

    return plane, holes


def apply_holes(
    plane: np.ndarray,
    holes: np.ndarray,
) -> np.ndarray:
    """Apply binary holes to a binary 2D plane."""

    plane = np.asarray(
        plane,
        dtype=np.uint8,
    )

    holes = np.asarray(
        holes,
        dtype=np.uint8,
    )

    if plane.ndim != 2:
        raise ValueError(
            "plane must be a 2D array"
        )

    if holes.ndim != 2:
        raise ValueError(
            "holes must be a 2D array"
        )

    if plane.shape != holes.shape:
        raise ValueError(
            "plane and holes must have the same shape"
        )

    if not np.all(
        np.isin(
            plane,
            [0, 1],
        ),
    ):
        raise ValueError(
            "plane must contain only 0 or 1"
        )

    if not np.all(
        np.isin(
            holes,
            [0, 1],
        ),
    ):
        raise ValueError(
            "holes must contain only 0 or 1"
        )

    return (
        plane * holes
    ).astype(
        np.uint8,
    )


def camera_exposure(
    plane: np.ndarray,
    holes: np.ndarray,
):
    """Apply holes and calculate the camera intensity."""

    transmitted_plane = apply_holes(
        plane,
        holes,
    )

    camera_field = fft2c(
        transmitted_plane,
    )

    camera_intensity = np.abs(
        camera_field,
    ) ** 2

    center_y = camera_field.shape[0] // 2
    center_x = camera_field.shape[1] // 2

    dc_index = (
        center_y,
        center_x,
    )

    dc_field = camera_field[dc_index]
    dc_intensity = float(
        camera_intensity[dc_index],
    )

    return {
        "plane": np.asarray(
            plane,
        ),
        "holes": np.asarray(
            holes,
        ),
        "transmitted_plane": transmitted_plane,
        "camera_field": camera_field,
        "camera_intensity": camera_intensity,
        "dc_index": dc_index,
        "dc_field": dc_field,
        "dc_intensity": dc_intensity,
        "dc_amplitude_magnitude": math.sqrt(
            dc_intensity,
        ),
    }


# =====================================================================
# BINARY OPERATIONS
# =====================================================================

def binary_multiply(
    x: int,
    y: int,
    return_optics: bool = False,
):
    """Multiply two binary values using one plane, holes, and camera."""

    if x not in (0, 1):
        raise ValueError(
            "x must be 0 or 1"
        )

    if y not in (0, 1):
        raise ValueError(
            "y must be 0 or 1"
        )

    plane = np.array(
        [[x]],
        dtype=np.uint8,
    )

    holes = np.array(
        [[y]],
        dtype=np.uint8,
    )

    optics = camera_exposure(
        plane,
        holes,
    )

    result = int(
        round(
            optics["dc_field"].real,
        ),
    )

    if return_optics:
        return result, optics

    return result


def binary_add_many(
    values,
    return_optics: bool = False,
):
    """Sum binary values using one binary plane and camera."""

    values = np.asarray(
        values,
        dtype=np.uint8,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional"
        )

    if not np.all(
        np.isin(
            values,
            [0, 1],
        ),
    ):
        raise ValueError(
            "values must contain only 0 or 1"
        )

    plane, holes = make_binary_plane_and_holes(
        values,
    )

    optics = camera_exposure(
        plane,
        holes,
    )

    result = int(
        round(
            optics["dc_field"].real,
        ),
    )

    if return_optics:
        return result, optics

    return result


# =====================================================================
# 3D BINARY LOOP
# =====================================================================

def evaluate_3d_binary_loop(
    A,
    B,
    C,
):
    """Evaluate a triple loop for binary input arrays."""

    A = np.asarray(
        A,
        dtype=np.uint8,
    )

    B = np.asarray(
        B,
        dtype=np.uint8,
    )

    C = np.asarray(
        C,
        dtype=np.uint8,
    )

    for values in (A, B, C):
        if values.ndim != 1:
            raise ValueError(
                "all inputs must be one-dimensional"
            )

        if not np.all(
            np.isin(
                values,
                [0, 1],
            ),
        ):
            raise ValueError(
                "all inputs must contain only 0 or 1"
            )

    terms = []
    trace = []

    for i, a in enumerate(A):
        for j, b in enumerate(B):
            for k, c in enumerate(C):

                ab = binary_multiply(
                    int(a),
                    int(b),
                )

                term = binary_multiply(
                    ab,
                    int(c),
                )

                terms.append(
                    term,
                )

                trace.append(
                    (
                        i,
                        j,
                        k,
                        int(a),
                        int(b),
                        int(c),
                        term,
                    ),
                )

    result, optics = binary_add_many(
        terms,
        return_optics=True,
    )

    return result, terms, trace, optics


# =====================================================================
# DIGITAL BASELINE
# =====================================================================

def digital_binary_loop(
    A,
    B,
    C,
):
    """Calculate the same binary triple loop digitally."""

    total = 0

    for a in A:
        for b in B:
            for c in C:
                total += int(a) * int(b) * int(c)

    return total


# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_matrix(
    fig,
    ax,
    data,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None,
    annotate=False,
    mark=None,
):
    """Plot a 2D matrix."""

    image = ax.imshow(
        data,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_title(
        title,
        fontsize=10,
    )

    if annotate and data.size <= 64:
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):

                value = data[row, column]

                if np.iscomplexobj(data):
                    label = f"{value.real:.0f}"
                else:
                    label = f"{value:.0f}"

                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )

    if mark is not None:
        mark_y, mark_x = mark

        ax.plot(
            mark_x,
            mark_y,
            marker="x",
            color="lime",
            markersize=12,
            markeredgewidth=2,
        )

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.75,
    )


def make_binary_visualization(
    plane,
    holes,
    optics,
    result,
    out_path,
):
    """Visualize the binary plane, holes, and camera."""

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        layout="constrained",
    )

    plot_matrix(
        fig,
        axes[0, 0],
        plane,
        "Binary 2D plane",
        cmap="gray",
        vmin=0,
        vmax=1,
        annotate=plane.size <= 64,
    )

    plot_matrix(
        fig,
        axes[0, 1],
        holes,
        "Binary holes",
        cmap="gray",
        vmin=0,
        vmax=1,
        annotate=holes.size <= 64,
    )

    plot_matrix(
        fig,
        axes[0, 2],
        optics["transmitted_plane"],
        "Transmitted binary plane",
        cmap="gray",
        vmin=0,
        vmax=1,
        annotate=optics["transmitted_plane"].size <= 64,
    )

    plot_matrix(
        fig,
        axes[1, 0],
        np.abs(
            optics["camera_field"],
        ),
        "Camera field magnitude",
        cmap="magma",
    )

    plot_matrix(
        fig,
        axes[1, 1],
        optics["camera_intensity"],
        "Camera intensity",
        cmap="inferno",
        mark=optics["dc_index"],
    )

    axes[1, 2].axis("off")

    lines = [
        "BINARY PLANE + HOLES + CAMERA",
        "",
        f"result: {result}",
        "",
        f"DC index: {optics['dc_index']}",
        f"DC field: {optics['dc_field']}",
        f"DC intensity: {optics['dc_intensity']:.3f}",
    ]

    axes[1, 2].text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=10,
        transform=axes[1, 2].transAxes,
    )

    fig.suptitle(
        "Binary 2D Plane, Holes, and Camera",
        fontsize=15,
    )

    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# DEMO
# =====================================================================

def run_demo(
    A=None,
    B=None,
    C=None,
    out_path="binary_plane_holes_camera.png",
    seed=7,
):
    """Run the binary-plane demonstration."""

    rng = random.Random(
        seed,
    )

    if A is None:
        A = [
            rng.randint(0, 1)
            for _ in range(4)
        ]

    if B is None:
        B = [
            rng.randint(0, 1)
            for _ in range(4)
        ]

    if C is None:
        C = [
            rng.randint(0, 1)
            for _ in range(4)
        ]

    start = time.perf_counter()

    digital_result = digital_binary_loop(
        A,
        B,
        C,
    )

    digital_time = time.perf_counter() - start

    start = time.perf_counter()

    optical_result, terms, trace, optics = (
        evaluate_3d_binary_loop(
            A,
            B,
            C,
        )
    )

    optical_time = time.perf_counter() - start

    plane, holes = make_binary_plane_and_holes(
        np.array(
            terms,
            dtype=np.uint8,
        ),
    )

    print("=" * 78)
    print("BINARY 2D PLANE, HOLES, AND CAMERA")
    print("=" * 78)
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"C: {C}")
    print(f"loop shape: {len(A)} × {len(B)} × {len(C)}")
    print(f"number of terms: {len(terms)}")
    print()
    print(f"digital result: {digital_result}")
    print(f"camera result: {optical_result}")
    print(f"match: {digital_result == optical_result}")
    print()
    print(f"2D plane shape: {plane.shape}")
    print(f"camera DC index: {optics['dc_index']}")
    print(f"camera DC intensity: {optics['dc_intensity']:.3f}")
    print()
    print(f"digital time: {digital_time * 1000:.3f} ms")
    print(f"camera simulation time: {optical_time * 1000:.3f} ms")

    if digital_result != optical_result:
        raise AssertionError(
            "camera result disagreed with digital result"
        )

    make_binary_visualization(
        plane,
        holes,
        optics,
        optical_result,
        out_path,
    )

    print(f"PNG: {out_path}")

    return optical_result, optics




"""
optical_eval.py
================

An "eval-for-code" front end for optical_binary.py.

The original module only exposes two primitives:

    binary_multiply(x, y)   -> one hole pair + camera read (AND)
    binary_add_many(values) -> one plane of holes + camera read (SUM)

Everything else (loops, De Morgan expansions, etc.) had to be hand-written
against those two calls. This module gives you two ways to *program*
against the optical backend instead of hand-wiring it:

1. Bit  - a tiny wrapper around 0/1 that overloads &, |, ^, ~ so you can
          just write ordinary Python boolean expressions and have every
          operation actually execute through camera_exposure().

2. qeval(expr, **vars) - a real "eval()" that takes a *string* of
          Python-flavoured boolean/bitwise syntax and runs it through the
          same optical primitives, returning the result (and, optionally,
          a step-by-step trace of every optical exposure that fired).

Both are built only from binary_multiply / binary_add_many, so nothing
here bypasses the "optics" - it's just a friendlier way to drive it.
"""

import ast

from optical_binary import binary_multiply, binary_add_many


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
# 1) Bit - operator-overloaded value, so plain Python syntax "is" the program
# =====================================================================

class Bit:
    """
    A 0/1 value whose operators are wired straight into the optical
    primitives. Write normal Python:

        a, b, c = Bit(1), Bit(0), Bit(1)
        result = (a & b) | c        # every &, |, ^, ~ below fires the camera

    `result.trace` accumulates a log of every optical exposure that ran,
    in order, so you can see exactly which hole-pairs/planes were used.
    """

    def __init__(self, value, trace=None):
        value = int(value)
        if value not in (0, 1):
            raise ValueError("Bit must be 0 or 1")
        self.value = value
        self.trace = trace if trace is not None else []

    def _merge_trace(self, other):
        merged = list(self.trace)
        if isinstance(other, Bit):
            merged.extend(other.trace)
        return merged

    def _log(self, op, inputs, result):
        self.trace_entry = (op, inputs, result)
        return self.trace_entry

    def _combine(self, other, op_name, op_fn):
        other_val = other.value if isinstance(other, Bit) else int(other)
        result = op_fn(self.value, other_val)
        trace = self._merge_trace(other)
        trace.append((op_name, (self.value, other_val), result))
        return Bit(result, trace=trace)

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
        if value not in (0, 1):
            raise ValueError(f"variable '{node.id}' must be 0 or 1")
        return value

    def visit_Constant(self, node):
        if node.value not in (0, 1):
            raise ValueError("literals must be 0 or 1")
        return int(node.value)

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in _BINOP_TABLE:
            raise SyntaxError(
                f"unsupported operator {op_type.__name__}; use & | ^ ~"
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

    Supported syntax: & | ^ ~  and the keywords and/or/not, plus
    parentheses and 0/1 variables - i.e. ordinary Python boolean syntax.
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
    print("""
    Prompt for an expression and its 0/1 variable values, then run it
    through ooeval(). Values are entered as "a=1,b=0,c=1".
 
    Example session:
        Enter expression: (a & b) | ~c
        Enter values: a=1,b=0,c=1
        ooeval('(a & b) | ~c', {'a': 1, 'b': 0, 'c': 1}) = 0
            [1] BitAnd(1, 0) -> 0
            [2] NOT(1,) -> 0
            [3] BitOr(0, 0) -> 0
    """)
 
    expr = input("Enter expression: ").strip()
    values_str = input("Enter values (e.g. a=1,b=0,c=1): ").strip()
 
    values = {}
    for pair in values_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"expected 'name=0/1', got '{pair}'"
            )
        name, val = pair.split("=", 1)
        name = name.strip()
        val = int(val.strip())
        if val not in (0, 1):
            raise ValueError(f"'{name}' must be 0 or 1, got {val}")
        values[name] = val
 
    result, trace = qeval(expr, return_trace=True, **values)
 
    print(f"\nooeval({expr!r}, {values}) = {result}")
    for step, (op, inputs, out) in enumerate(trace, start=1):
        print(f"    [{step}] {op}{inputs} -> {out}")
 
    return result
if __name__ == "__main__":

    print()
    print("=" * 78)
    print("qeval - text expressions evaluated through the optical backend")
    print("=" * 78)
    interactive_eval()
