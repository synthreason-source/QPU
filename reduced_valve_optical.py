#!/usr/bin/env python3
"""
Reduced-valve optical Q compiler/interpreter.

Features
--------
- Time-multiplexed binary optical operations
- 32-bit arithmetic
- AND32 / OR32 / XOR32 / NOT32
- ADD32
- SHR / SHL / ROTR
- Lists
- List indexing
- List assignment
- Functions
- Return
- If / else
- While
- For / range
- SHA-256 capable Q programs
- Optical operation tracing
- Optical statistics
- Simulator backend

Example:

    python reduced_valve_optical_stats.py \
        --program sha256.q \
        --input abc

Self-test:

    python reduced_valve_optical_stats.py \
        --program sha256.q \
        --self-test

Trace:

    python reduced_valve_optical_stats.py \
        --program sha256.q \
        --input abc \
        --trace \
        --trace-limit 100

Stats:

    python reduced_valve_optical_stats.py \
        --program sha256.q \
        --input abc \
        --stats
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# ============================================================
# CONSTANTS
# ============================================================

MAX_LOOP_ITERATIONS = 1_000_000
MASK32 = 0xFFFFFFFF


# ============================================================
# VALIDATION
# ============================================================

def _nonnegative_int(
    name: str,
    value: Any,
) -> int:

    if isinstance(value, bool):
        raise ValueError(
            f"{name} expects an integer"
        )

    if not isinstance(value, int):
        raise ValueError(
            f"{name} expects an integer, "
            f"got {value!r}"
        )

    if value < 0:
        raise ValueError(
            f"{name} expects a nonnegative "
            f"integer, got {value!r}"
        )

    return value


def _bit(
    name: str,
    value: Any,
) -> int:

    value = _nonnegative_int(
        name,
        value,
    )

    if value not in (0, 1):
        raise ValueError(
            f"{name} expects 0 or 1, "
            f"got {value!r}"
        )

    return value


def _u32(
    value: int,
) -> int:

    return int(value) & MASK32


# ============================================================
# BIT CONVERSION
# ============================================================

def int_to_bits(
    value: int,
    width: Optional[int] = None,
) -> list[int]:

    value = _nonnegative_int(
        "value",
        value,
    )

    if width is None:
        width = max(
            1,
            value.bit_length(),
        )

    if width < 1:
        raise ValueError(
            "width must be positive"
        )

    if value >= (1 << width):
        raise ValueError(
            f"{value} does not fit "
            f"in {width} bits"
        )

    return [
        (value >> i) & 1
        for i in range(width)
    ]


def bits_to_int(
    bits: Iterable[int],
) -> int:

    result = 0

    for i, value in enumerate(bits):

        result |= (
            _bit(
                "bit",
                int(value),
            )
            << i
        )

    return result


# ============================================================
# OPTICAL SIMULATOR
# ============================================================

class SimulatedOptics:
    """
    Deterministic optical hardware simulator.

    The two-valve concept is represented by reusable bit
    operations rather than unary hole-count arithmetic.
    """

    def and_bit(
        self,
        a: int,
        b: int,
    ) -> int:

        return (
            _bit("a", a)
            &
            _bit("b", b)
        )

    def sum_bits(
        self,
        values: Iterable[int],
    ) -> int:

        return sum(
            _bit(
                "value",
                int(v),
            )
            for v in values
        )

    def sample(
        self,
        value: int,
    ) -> int:

        return int(value)


# ============================================================
# OPTICAL BACKEND
# ============================================================

class OpticalBackend:

    def __init__(
        self,
        driver: Optional[Any] = None,
        valve_count: int = 2,
    ):

        self.driver = (
            driver
            or SimulatedOptics()
        )

        self.valve_count = valve_count

    def and_bit(
        self,
        a: int,
        b: int,
    ) -> int:

        return _bit(
            "AND result",
            self.driver.and_bit(
                a,
                b,
            ),
        )

    def sum_bits(
        self,
        values: Iterable[int],
    ) -> int:

        return int(
            self.driver.sum_bits(values)
        )

    def sample(
        self,
        value: int,
    ) -> int:

        return int(
            self.driver.sample(value)
        )


# ============================================================
# OPTICAL TRACE
# ============================================================

@dataclass
class OpticalTrace:

    events: list[
        tuple[
            str,
            tuple[Any, ...],
            Any,
        ]
    ] = field(
        default_factory=list
    )

    def record(
        self,
        operation: str,
        inputs: tuple[Any, ...],
        result: Any,
    ) -> Any:

        self.events.append(
            (
                operation,
                inputs,
                result,
            )
        )

        return result

    def print(
        self,
        limit: Optional[int] = None,
    ) -> None:

        events = self.events

        if limit is not None:
            events = events[:limit]

        for index, (
            operation,
            inputs,
            result,
        ) in enumerate(
            events,
            1,
        ):

            print(
                f"[{index}] "
                f"{operation}{inputs} "
                f"-> {result}"
            )

        if (
            limit is not None
            and len(self.events) > limit
        ):

            print(
                f"... "
                f"{len(self.events) - limit} "
                f"more events"
            )


# ============================================================
# STATISTICS
# ============================================================

@dataclass
class OpticalStats:

    counts: dict[str, int] = field(
        default_factory=dict
    )

    total_events: int = 0

    def record(
        self,
        operation: str,
    ) -> None:

        self.total_events += 1

        self.counts[operation] = (
            self.counts.get(
                operation,
                0,
            )
            + 1
        )

    def print(self) -> None:

        print()
        print("=" * 60)
        print("OPTICAL STATISTICS")
        print("=" * 60)

        print(
            f"Total optical events: "
            f"{self.total_events}"
        )

        print()

        for name, count in sorted(
            self.counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):

            print(
                f"{name:25s} "
                f"{count:12d}"
            )

        print("=" * 60)


# ============================================================
# REDUCED VALVE MACHINE
# ============================================================

class ReducedValveMachine:

    def __init__(
        self,
        backend: Optional[
            OpticalBackend
        ] = None,
        width: int = 32,
        trace: Optional[
            OpticalTrace
        ] = None,
        stats: Optional[
            OpticalStats
        ] = None,
    ):

        self.backend = (
            backend
            or OpticalBackend()
        )

        self.width = width

        self.trace = (
            trace
            or OpticalTrace()
        )

        self.stats = (
            stats
            or OpticalStats()
        )

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    def _record(
        self,
        operation: str,
        inputs: tuple[Any, ...],
        result: Any,
    ) -> Any:

        self.stats.record(
            operation
        )

        return self.trace.record(
            operation,
            inputs,
            result,
        )

    # --------------------------------------------------------
    # BIT OPERATIONS
    # --------------------------------------------------------

    def and_bit(
        self,
        a: int,
        b: int,
    ) -> int:

        result = self.backend.and_bit(
            _bit("a", a),
            _bit("b", b),
        )

        return self._record(
            "AND",
            (a, b),
            result,
        )

    def or_bit(
        self,
        a: int,
        b: int,
    ) -> int:

        total = self.backend.sum_bits(
            [
                _bit("a", a),
                _bit("b", b),
            ]
        )

        result = int(total > 0)

        return self._record(
            "OR",
            (a, b),
            result,
        )

    def xor_bit(
        self,
        a: int,
        b: int,
    ) -> int:

        total = self.backend.sum_bits(
            [
                _bit("a", a),
                _bit("b", b),
            ]
        )

        result = total & 1

        return self._record(
            "XOR",
            (a, b),
            result,
        )

    def not_bit(
        self,
        a: int,
    ) -> int:

        result = 1 - _bit(
            "NOT",
            a,
        )

        return self._record(
            "NOT",
            (a,),
            result,
        )

    # --------------------------------------------------------
    # SERIAL BINARY
    # --------------------------------------------------------

    def _serial_binary(
        self,
        a: int,
        b: int,
        operation: str,
        width: Optional[int] = None,
    ) -> int:

        width = (
            width
            or self.width
        )

        a = _u32(a)
        b = _u32(b)

        abits = int_to_bits(
            a,
            width,
        )

        bbits = int_to_bits(
            b,
            width,
        )

        output = []

        for slot, (
            ai,
            bi,
        ) in enumerate(
            zip(
                abits,
                bbits,
            )
        ):

            if operation == "AND":

                value = (
                    self.backend.and_bit(
                        ai,
                        bi,
                    )
                )

            elif operation == "OR":

                value = int(
                    self.backend.sum_bits(
                        [ai, bi]
                    ) > 0
                )

            elif operation == "XOR":

                value = (
                    self.backend.sum_bits(
                        [ai, bi]
                    )
                    & 1
                )

            else:

                raise ValueError(
                    f"unknown operation "
                    f"{operation}"
                )

            output.append(
                value
            )

            self._record(
                f"{operation}[t={slot}]",
                (ai, bi),
                value,
            )

        return bits_to_int(
            output
        )

    # --------------------------------------------------------
    # 32 BIT LOGIC
    # --------------------------------------------------------

    def and32(
        self,
        *values: int,
    ) -> int:

        if not values:
            return MASK32

        result = _u32(
            values[0]
        )

        for value in values[1:]:

            result = (
                self._serial_binary(
                    result,
                    _u32(value),
                    "AND",
                    32,
                )
                & MASK32
            )

        return self._record(
            "AND32",
            tuple(values),
            result,
        )

    def or32(
        self,
        *values: int,
    ) -> int:

        if not values:
            return 0

        result = _u32(
            values[0]
        )

        for value in values[1:]:

            result = (
                self._serial_binary(
                    result,
                    _u32(value),
                    "OR",
                    32,
                )
                & MASK32
            )

        return self._record(
            "OR32",
            tuple(values),
            result,
        )

    def xor32(
        self,
        *values: int,
    ) -> int:

        if not values:
            return 0

        result = _u32(
            values[0]
        )

        for value in values[1:]:

            result = (
                self._serial_binary(
                    result,
                    _u32(value),
                    "XOR",
                    32,
                )
                & MASK32
            )

        return self._record(
            "XOR32",
            tuple(values),
            result,
        )

    def not32(
        self,
        value: int,
    ) -> int:

        result = (
            ~_u32(value)
        ) & MASK32

        return self._record(
            "NOT32",
            (value,),
            result,
        )

    # --------------------------------------------------------
    # ADDITION
    # --------------------------------------------------------

    def _add_raw(
        self,
        a: int,
        b: int,
    ) -> int:

        carry = 0
        result = 0

        for slot in range(32):

            ai = (
                a >> slot
            ) & 1

            bi = (
                b >> slot
            ) & 1

            total = (
                self.backend.sum_bits(
                    [
                        ai,
                        bi,
                        carry,
                    ]
                )
            )

            output_bit = (
                total & 1
            )

            carry = int(
                total >= 2
            )

            result |= (
                output_bit
                << slot
            )

            self._record(
                f"ADD32[t={slot}]",
                (
                    ai,
                    bi,
                    carry,
                ),
                output_bit,
            )

        return result & MASK32

    def add32(
        self,
        *values: int,
    ) -> int:

        result = 0

        for value in values:

            result = self._add_raw(
                result,
                _u32(value),
            )

        result &= MASK32

        return self._record(
            "ADD32",
            tuple(values),
            result,
        )

    # --------------------------------------------------------
    # SUBTRACT
    # --------------------------------------------------------

    def subtract(
        self,
        a: int,
        b: int,
    ) -> int:

        result = (
            _u32(a)
            -
            _u32(b)
        ) & MASK32

        return self._record(
            "SUB32",
            (a, b),
            result,
        )

    # --------------------------------------------------------
    # SHIFTS
    # --------------------------------------------------------

    def shr(
        self,
        value: int,
        amount: int,
    ) -> int:

        value = _u32(value)
        amount = int(amount)

        if amount >= 32:
            result = 0
        else:
            result = (
                value >> amount
            )

        return self._record(
            "SHR",
            (value, amount),
            result,
        )

    def shl(
        self,
        value: int,
        amount: int,
    ) -> int:

        result = (
            _u32(value)
            << int(amount)
        ) & MASK32

        return self._record(
            "SHL",
            (value, amount),
            result,
        )

    def rotr(
        self,
        value: int,
        amount: int,
    ) -> int:

        value = _u32(value)

        amount %= 32

        if amount == 0:

            result = value

        else:

            result = (
                (value >> amount)
                |
                (
                    value
                    << (32 - amount)
                )
            ) & MASK32

        return self._record(
            "ROTR",
            (value, amount),
            result,
        )

    # --------------------------------------------------------
    # MULTIPLY
    # --------------------------------------------------------

    def multiply(
        self,
        a: int,
        b: int,
    ) -> int:

        result = 0
        row = _u32(a)

        b = _u32(b)

        shift = 0

        while (
            b >> shift
        ):

            if (
                (b >> shift) & 1
            ):

                result = self.add32(
                    result,
                    row,
                )

            row = (
                row << 1
            ) & MASK32

            shift += 1

        return self._record(
            "MUL32",
            (a, b),
            result,
        )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    def compare(
        self,
        a: int,
        b: int,
    ) -> int:

        if a == b:
            result = 0

        elif a > b:
            result = 1

        else:
            result = -1

        return self._record(
            "CMP",
            (a, b),
            result,
        )

    def eq(self, a, b):
        return int(
            self.compare(a, b) == 0
        )

    def ne(self, a, b):
        return int(
            self.compare(a, b) != 0
        )

    def lt(self, a, b):
        return int(
            self.compare(a, b) < 0
        )

    def le(self, a, b):
        return int(
            self.compare(a, b) <= 0
        )

    def gt(self, a, b):
        return int(
            self.compare(a, b) > 0
        )

    def ge(self, a, b):
        return int(
            self.compare(a, b) >= 0
        )


# ============================================================
# Q RUNTIME
# ============================================================

class QRuntime:

    def __init__(
        self,
        machine: ReducedValveMachine,
        data: bytes,
    ):

        self.machine = machine
        self.data = bytes(data)

    def byte(
        self,
        index: int,
    ) -> int:

        index = int(index)

        if (
            index < 0
            or index >= len(self.data)
        ):

            return 0

        return self.data[index]

    def length(self) -> int:

        return len(self.data)

    def padded(
        self,
    ) -> bytes:

        message = bytearray(
            self.data
        )

        bit_length = (
            len(message)
            * 8
        )

        message.append(
            0x80
        )

        while (
            len(message) % 64
            != 56
        ):

            message.append(
                0
            )

        message.extend(
            bit_length.to_bytes(
                8,
                "big",
            )
        )

        return bytes(message)

    def word(
        self,
        index: int,
    ) -> int:

        data = self.padded()

        offset = (
            int(index)
            * 4
        )

        if (
            offset + 4
            > len(data)
        ):

            return 0

        return int.from_bytes(
            data[
                offset:
                offset + 4
            ],
            "big",
        )

    def word_count(
        self,
    ) -> int:

        return (
            len(self.padded())
            // 4
        )


# ============================================================
# Q CONTROL EXCEPTIONS
# ============================================================

class QReturn(Exception):

    def __init__(
        self,
        value: Any,
    ):

        self.value = value


class QBreak(Exception):
    pass


class QContinue(Exception):
    pass


# ============================================================
# Q FUNCTIONS
# ============================================================

@dataclass
class QFunction:

    node: ast.FunctionDef


# ============================================================
# EXPRESSION EVALUATOR
# ============================================================

class ExpressionEvaluator(
    ast.NodeVisitor
):

    def __init__(
        self,
        machine: ReducedValveMachine,
        variables: dict[str, Any],
        functions: dict[str, QFunction],
        runtime: QRuntime,
        executor: "StatementExecutor",
    ):

        self.machine = machine
        self.variables = variables
        self.functions = functions
        self.runtime = runtime
        self.executor = executor

    # --------------------------------------------------------
    # CONSTANT
    # --------------------------------------------------------

    def visit_Constant(
        self,
        node: ast.Constant,
    ):

        if isinstance(
            node.value,
            (
                int,
                str,
                bytes,
            ),
        ):

            return node.value

        if node.value is None:

            return None

        raise SyntaxError(
            f"unsupported constant: "
            f"{node.value!r}"
        )

    # --------------------------------------------------------
    # VARIABLE
    # --------------------------------------------------------

    def visit_Name(
        self,
        node: ast.Name,
    ):

        if (
            node.id
            not in self.variables
        ):

            raise NameError(
                f"undefined Q variable: "
                f"{node.id}"
            )

        return self.variables[
            node.id
        ]

    # --------------------------------------------------------
    # LIST
    #
    # THIS FIXES THE ERROR IN YOUR TRACEBACK.
    # --------------------------------------------------------

    def visit_List(
        self,
        node: ast.List,
    ):

        return [
            self.visit(element)
            for element in node.elts
        ]

    # --------------------------------------------------------
    # TUPLE
    # --------------------------------------------------------

    def visit_Tuple(
        self,
        node: ast.Tuple,
    ):

        return tuple(
            self.visit(element)
            for element in node.elts
        )

    # --------------------------------------------------------
    # INDEXING
    #
    # Supports:
    #
    #   K[j]
    #   w[j]
    #   state[0]
    # --------------------------------------------------------

    def visit_Subscript(
        self,
        node: ast.Subscript,
    ):

        container = self.visit(
            node.value
        )

        index = self.visit(
            node.slice
        )

        return container[index]

    # Python 3.11 compatibility.

    def visit_Index(
        self,
        node: ast.Index,
    ):

        return self.visit(
            node.value
        )

    # --------------------------------------------------------
    # LIST REPETITION
    #
    # Supports:
    #
    #   [0] * 64
    # --------------------------------------------------------

    # --------------------------------------------------------
    # UNARY
    # --------------------------------------------------------

    def visit_UnaryOp(
        self,
        node: ast.UnaryOp,
    ):

        value = self.visit(
            node.operand
        )

        if isinstance(
            node.op,
            ast.Invert,
        ):

            return self.machine.not32(
                value
            )

        if isinstance(
            node.op,
            ast.USub,
        ):

            return (
                -int(value)
            ) & MASK32

        if isinstance(
            node.op,
            ast.UAdd,
        ):

            return int(value)

        if isinstance(
            node.op,
            ast.Not,
        ):

            return int(
                not value
            )

        raise SyntaxError(
            "unsupported unary operation"
        )

    # --------------------------------------------------------
    # BINARY
    # --------------------------------------------------------

    def visit_BinOp(
        self,
        node: ast.BinOp,
    ):

        left = self.visit(
            node.left
        )

        right = self.visit(
            node.right
        )

        # List repetition.

        if isinstance(
            node.op,
            ast.Mult,
        ):

            if isinstance(
                left,
                list,
            ) and isinstance(
                right,
                int,
            ):

                return left * right

            if isinstance(
                right,
                list,
            ) and isinstance(
                left,
                int,
            ):

                return right * left

        # String/list concatenation.

        if isinstance(
            node.op,
            ast.Add,
        ) and (
            isinstance(left, list)
            or isinstance(right, list)
            or isinstance(left, str)
            or isinstance(right, str)
        ):

            return left + right

        if isinstance(
            node.op,
            ast.BitAnd,
        ):

            return self.machine.and32(
                left,
                right,
            )

        if isinstance(
            node.op,
            ast.BitOr,
        ):

            return self.machine.or32(
                left,
                right,
            )

        if isinstance(
            node.op,
            ast.BitXor,
        ):

            return self.machine.xor32(
                left,
                right,
            )

        if isinstance(
            node.op,
            ast.Add,
        ):

            return self.machine.add32(
                left,
                right,
            )

        if isinstance(
            node.op,
            ast.Sub,
        ):

            return self.machine.subtract(
                left,
                right,
            )

        if isinstance(
            node.op,
            ast.LShift,
        ):

            return self.machine.shl(
                left,
                right,
            )

        if isinstance(
            node.op,
            ast.RShift,
        ):

            return self.machine.shr(
                left,
                right,
            )

        if isinstance(
            node.op,
            ast.Mod,
        ):

            return left % right

        if isinstance(
            node.op,
            ast.Mult,
        ):

            return left * right

        raise SyntaxError(
            "unsupported binary operator"
        )

    # --------------------------------------------------------
    # BOOLEAN
    # --------------------------------------------------------

    def visit_BoolOp(
        self,
        node: ast.BoolOp,
    ):

        values = [
            self.visit(v)
            for v in node.values
        ]

        if isinstance(
            node.op,
            ast.And,
        ):

            return int(
                all(values)
            )

        if isinstance(
            node.op,
            ast.Or,
        ):

            return int(
                any(values)
            )

        raise SyntaxError(
            "unsupported boolean operator"
        )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    def visit_Compare(
        self,
        node: ast.Compare,
    ):

        left = self.visit(
            node.left
        )

        result = True

        for op, comparator in zip(
            node.ops,
            node.comparators,
        ):

            right = self.visit(
                comparator
            )

            if isinstance(
                op,
                ast.Eq,
            ):

                current = (
                    left == right
                )

            elif isinstance(
                op,
                ast.NotEq,
            ):

                current = (
                    left != right
                )

            elif isinstance(
                op,
                ast.Lt,
            ):

                current = (
                    left < right
                )

            elif isinstance(
                op,
                ast.LtE,
            ):

                current = (
                    left <= right
                )

            elif isinstance(
                op,
                ast.Gt,
            ):

                current = (
                    left > right
                )

            elif isinstance(
                op,
                ast.GtE,
            ):

                current = (
                    left >= right
                )

            else:

                raise SyntaxError(
                    "unsupported comparison"
                )

            result = (
                result
                and current
            )

            left = right

        return int(result)

    # --------------------------------------------------------
    # CALL
    # --------------------------------------------------------

    def visit_Call(
        self,
        node: ast.Call,
    ):

        if not isinstance(
            node.func,
            ast.Name,
        ):

            raise SyntaxError(
                "Q only permits "
                "named function calls"
            )

        name = node.func.id

        args = [
            self.visit(arg)
            for arg in node.args
        ]

        # Optical primitives.

        if name == "ADD32":

            return self.machine.add32(
                *args
            )

        if name == "AND32":

            return self.machine.and32(
                *args
            )

        if name == "OR32":

            return self.machine.or32(
                *args
            )

        if name == "XOR32":

            return self.machine.xor32(
                *args
            )

        if name == "NOT32":

            return self.machine.not32(
                args[0]
            )

        if name == "ROTR":

            return self.machine.rotr(
                args[0],
                args[1],
            )

        if name == "SHR":

            return self.machine.shr(
                args[0],
                args[1],
            )

        if name == "SHL":

            return self.machine.shl(
                args[0],
                args[1],
            )

        # Runtime input.

        if name == "BYTE":

            return self.runtime.byte(
                args[0]
            )

        if name == "INPUT_LENGTH":

            return self.runtime.length()

        if name == "INPUT_WORD":

            return self.runtime.word(
                args[0]
            )

        if name == "WORD_COUNT":

            return self.runtime.word_count()

        # Helpers.

        if name == "len":

            return len(args[0])

        if name == "range":

            return range(*args)

        if name == "int":

            return int(args[0])

        if name == "U32":

            return _u32(
                args[0]
            )

        # User function.

        if name in self.functions:

            return self.executor.call_function(
                self.functions[name],
                args,
            )

        raise NameError(
            f"unknown Q function: "
            f"{name}"
        )

    # --------------------------------------------------------
    # OTHERWISE
    # --------------------------------------------------------

    def generic_visit(
        self,
        node,
    ):

        raise SyntaxError(
            f"unsupported syntax: "
            f"{type(node).__name__}"
        )


# ============================================================
# STATEMENT EXECUTOR
# ============================================================

class StatementExecutor(
    ast.NodeVisitor
):

    def __init__(
        self,
        machine: ReducedValveMachine,
        variables: Optional[
            dict[str, Any]
        ] = None,
        runtime: Optional[
            QRuntime
        ] = None,
        functions: Optional[
            dict[str, QFunction]
        ] = None,
    ):

        self.machine = machine

        self.variables = (
            variables
            if variables is not None
            else {}
        )

        self.runtime = runtime

        self.functions = (
            functions
            if functions is not None
            else {}
        )

    def evaluate(
        self,
        node,
    ):

        return ExpressionEvaluator(
            self.machine,
            self.variables,
            self.functions,
            self.runtime,
            self,
        ).visit(node)

    # --------------------------------------------------------
    # MODULE
    # --------------------------------------------------------

    def visit_Module(
        self,
        node: ast.Module,
    ):

        for statement in node.body:

            self.visit(
                statement
            )

    # --------------------------------------------------------
    # FUNCTION
    # --------------------------------------------------------

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ):

        self.functions[
            node.name
        ] = QFunction(node)

    # --------------------------------------------------------
    # FUNCTION CALL
    # --------------------------------------------------------

    def call_function(
        self,
        function: QFunction,
        args: list[Any],
    ):

        node = function.node

        expected = len(
            node.args.args
        )

        if len(args) != expected:

            raise TypeError(
                f"{node.name} expects "
                f"{expected} arguments, "
                f"got {len(args)}"
            )

        local = dict(
            self.variables
        )

        for argument, value in zip(
            node.args.args,
            args,
        ):

            local[
                argument.arg
            ] = value

        child = StatementExecutor(
            self.machine,
            local,
            self.runtime,
            self.functions,
        )

        try:

            for statement in node.body:

                child.visit(
                    statement
                )

        except QReturn as returned:

            return returned.value

        return None

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    def visit_Return(
        self,
        node: ast.Return,
    ):

        value = (
            self.evaluate(
                node.value
            )
            if node.value is not None
            else None
        )

        raise QReturn(
            value
        )

    # --------------------------------------------------------
    # ASSIGNMENT
    # --------------------------------------------------------

    def visit_Assign(
        self,
        node: ast.Assign,
    ):

        value = self.evaluate(
            node.value
        )

        for target in node.targets:

            self.assign(
                target,
                value,
            )

    # --------------------------------------------------------
    # ASSIGN TARGET
    # --------------------------------------------------------

    def assign(
        self,
        target,
        value,
    ):

        # x = value

        if isinstance(
            target,
            ast.Name,
        ):

            self.variables[
                target.id
            ] = value

            return

        # w[j] = value

        if isinstance(
            target,
            ast.Subscript,
        ):

            container = self.evaluate(
                target.value
            )

            index = self.evaluate(
                target.slice
            )

            container[
                index
            ] = value

            return

        raise SyntaxError(
            "unsupported assignment target"
        )

    # --------------------------------------------------------
    # AUGMENTED ASSIGNMENT
    # --------------------------------------------------------

    def visit_AugAssign(
        self,
        node: ast.AugAssign,
    ):

        # Simple variable +=.

        if isinstance(
            node.target,
            ast.Name,
        ):

            name = node.target.id

            lhs = self.variables[
                name
            ]

            rhs = self.evaluate(
                node.value
            )

            if isinstance(
                node.op,
                ast.Add,
            ):

                value = (
                    self.machine.add32(
                        lhs,
                        rhs,
                    )
                )

            elif isinstance(
                node.op,
                ast.Sub,
            ):

                value = (
                    self.machine.subtract(
                        lhs,
                        rhs,
                    )
                )

            elif isinstance(
                node.op,
                ast.BitXor,
            ):

                value = (
                    self.machine.xor32(
                        lhs,
                        rhs,
                    )
                )

            elif isinstance(
                node.op,
                ast.BitAnd,
            ):

                value = (
                    self.machine.and32(
                        lhs,
                        rhs,
                    )
                )

            elif isinstance(
                node.op,
                ast.BitOr,
            ):

                value = (
                    self.machine.or32(
                        lhs,
                        rhs,
                    )
                )

            else:

                raise SyntaxError(
                    "unsupported "
                    "augmented operator"
                )

            self.variables[
                name
            ] = value

            return

        # w[j] += value

        if isinstance(
            node.target,
            ast.Subscript,
        ):

            container = self.evaluate(
                node.target.value
            )

            index = self.evaluate(
                node.target.slice
            )

            lhs = container[index]

            rhs = self.evaluate(
                node.value
            )

            if isinstance(
                node.op,
                ast.Add,
            ):

                container[index] = (
                    self.machine.add32(
                        lhs,
                        rhs,
                    )
                )

                return

            if isinstance(
                node.op,
                ast.BitXor,
            ):

                container[index] = (
                    self.machine.xor32(
                        lhs,
                        rhs,
                    )
                )

                return

            raise SyntaxError(
                "unsupported indexed "
                "augmented operator"
            )

        raise SyntaxError(
            "unsupported augmented "
            "assignment"
        )

    # --------------------------------------------------------
    # EXPRESSION STATEMENT
    # --------------------------------------------------------

    def visit_Expr(
        self,
        node: ast.Expr,
    ):

        if isinstance(
            node.value,
            ast.Call,
        ):

            if (
                isinstance(
                    node.value.func,
                    ast.Name,
                )
                and node.value.func.id
                == "print"
            ):

                values = [
                    self.evaluate(
                        arg
                    )
                    for arg
                    in node.value.args
                ]

                print(
                    *values
                )

                return

            if (
                isinstance(
                    node.value.func,
                    ast.Name,
                )
                and node.value.func.id
                == "print_hex"
            ):

                value = self.evaluate(
                    node.value.args[0]
                )

                print(
                    f"{_u32(value):08x}"
                )

                return

            if (
                isinstance(
                    node.value.func,
                    ast.Name,
                )
                and node.value.func.id
                == "print_digest"
            ):

                values = [
                    self.evaluate(
                        arg
                    )
                    for arg
                    in node.value.args
                ]

                print(
                    "".join(
                        f"{_u32(value):08x}"
                        for value
                        in values
                    )
                )

                return

        self.evaluate(
            node.value
        )

    # --------------------------------------------------------
    # IF
    # --------------------------------------------------------

    def visit_If(
        self,
        node: ast.If,
    ):

        if self.evaluate(
            node.test
        ):

            for statement in node.body:

                self.visit(
                    statement
                )

        else:

            for statement in node.orelse:

                self.visit(
                    statement
                )

    # --------------------------------------------------------
    # WHILE
    # --------------------------------------------------------

    def visit_While(
        self,
        node: ast.While,
    ):

        iterations = 0

        while self.evaluate(
            node.test
        ):

            iterations += 1

            if (
                iterations
                > MAX_LOOP_ITERATIONS
            ):

                raise RuntimeError(
                    "Q loop limit exceeded"
                )

            try:

                for statement in node.body:

                    self.visit(
                        statement
                    )

            except QBreak:

                break

            except QContinue:

                continue

    # --------------------------------------------------------
    # FOR
    # --------------------------------------------------------

    def visit_For(
        self,
        node: ast.For,
    ):

        if not isinstance(
            node.target,
            ast.Name,
        ):

            raise SyntaxError(
                "Q for target must "
                "be a variable"
            )

        iterable = self.evaluate(
            node.iter
        )

        for value in iterable:

            self.variables[
                node.target.id
            ] = value

            try:

                for statement in node.body:

                    self.visit(
                        statement
                    )

            except QBreak:

                break

            except QContinue:

                continue

    # --------------------------------------------------------
    # BREAK
    # --------------------------------------------------------

    def visit_Break(
        self,
        node: ast.Break,
    ):

        raise QBreak()

    # --------------------------------------------------------
    # CONTINUE
    # --------------------------------------------------------

    def visit_Continue(
        self,
        node: ast.Continue,
    ):

        raise QContinue()

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    def visit_Pass(
        self,
        node: ast.Pass,
    ):

        pass

    # --------------------------------------------------------
    # UNSUPPORTED
    # --------------------------------------------------------

    def generic_visit(
        self,
        node,
    ):

        raise SyntaxError(
            f"unsupported Q statement: "
            f"{type(node).__name__}"
        )


# ============================================================
# Q VALIDATOR
# ============================================================

FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.With,
    ast.Try,
    ast.Raise,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
    ast.Yield,
    ast.YieldFrom,
)


BUILTIN_CALLS = {
    "ADD32",
    "AND32",
    "OR32",
    "XOR32",
    "NOT32",
    "ROTR",
    "SHR",
    "SHL",
    "BYTE",
    "INPUT_LENGTH",
    "INPUT_WORD",
    "WORD_COUNT",
    "U32",
    "len",
    "range",
    "int",
    "print",
    "print_hex",
    "print_digest",
}


def validate_q(
    tree: ast.AST,
) -> None:

    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.FunctionDef,
        )
    }

    allowed_calls = (
        BUILTIN_CALLS
        | functions
    )

    for node in ast.walk(tree):

        if isinstance(
            node,
            FORBIDDEN_NODES,
        ):

            raise SyntaxError(
                "forbidden Q construct: "
                f"{type(node).__name__}"
            )

        if isinstance(
            node,
            ast.Call,
        ):

            if not isinstance(
                node.func,
                ast.Name,
            ):

                raise SyntaxError(
                    "Q requires named "
                    "function calls"
                )

            if (
                node.func.id
                not in allowed_calls
            ):

                raise SyntaxError(
                    f"unknown Q function: "
                    f"{node.func.id}"
                )


# ============================================================
# Q RUNNER
# ============================================================

def run_q(
    source: str,
    data: bytes,
    trace: Optional[
        OpticalTrace
    ] = None,
    statistics: Optional[
        OpticalStats
    ] = None,
):

    machine = ReducedValveMachine(
        width=32,
        trace=(
            trace
            or OpticalTrace()
        ),
        stats=(
            statistics
            or OpticalStats()
        ),
    )

    runtime = QRuntime(
        machine,
        data,
    )

    variables = {}

    tree = ast.parse(
        source,
        filename="<Q>",
        mode="exec",
    )

    validate_q(
        tree
    )

    executor = StatementExecutor(
        machine,
        variables,
        runtime,
    )

    executor.visit(
        tree
    )

    return (
        variables,
        machine,
    )


# ============================================================
# SHA-256 RESULT
# ============================================================

def extract_sha256(
    variables: dict[str, Any],
) -> str:

    names = (
        "h0",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "h7",
    )

    missing = [
        name
        for name in names
        if name not in variables
    ]

    if missing:

        raise RuntimeError(
            "Q program did not produce "
            "SHA-256 state variables: "
            + ", ".join(missing)
        )

    return "".join(
        f"{_u32(variables[name]):08x}"
        for name in names
    )


# ============================================================
# SELF TEST
# ============================================================

def self_test(
    source: str,
) -> None:

    tests = [
        b"",
        b"a",
        b"abc",
        b"hello",
        b"hello world",
        b"The quick brown fox jumps over the lazy dog",
        b"The quick brown fox jumps over the lazy dog.",
        b"a" * 55,
        b"a" * 56,
        b"a" * 57,
        b"a" * 63,
        b"a" * 64,
        b"a" * 65,
        b"a" * 127,
        b"a" * 128,
        b"a" * 129,
        b"a" * 1000,
    ]

    for data in tests:

        variables, machine = run_q(
            source,
            data,
        )

        got = extract_sha256(
            variables
        )

        expected = hashlib.sha256(
            data
        ).hexdigest()

        if got != expected:

            print()
            print(
                "SHA-256 FAILURE"
            )

            print(
                f"length : {len(data)}"
            )

            print(
                f"Q      : {got}"
            )

            print(
                f"hashlib: {expected}"
            )

            raise SystemExit(1)

        print(
            f"PASS "
            f"{len(data):5d} bytes "
            f"{got}"
        )

    print()
    print(
        "ALL Q SHA-256 TESTS PASSED"
    )


# ============================================================
# INPUT
# ============================================================

def load_input(
    args,
) -> bytes:

    if args.input is not None:

        return args.input.encode(
            "utf-8"
        )

    if args.file:

        with open(
            args.file,
            "rb",
        ) as f:

            return f.read()

    if args.hex is not None:

        return bytes.fromhex(
            args.hex
        )

    if args.stdin:

        return sys.stdin.buffer.read()

    return b""


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Reduced-valve optical Q compiler"
        )
    )

    parser.add_argument(
        "--program",
        "-p",
        required=True,
        help="Q source file",
    )

    parser.add_argument(
        "--input",
        help="UTF-8 input string",
    )

    parser.add_argument(
        "--file",
        "-f",
        help="binary input file",
    )

    parser.add_argument(
        "--hex",
        help="hexadecimal input",
    )

    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read input from stdin",
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help="print optical trace",
    )

    parser.add_argument(
        "--trace-limit",
        type=int,
        default=100,
        help="maximum trace events displayed",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="print optical statistics",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="test Q SHA-256 against hashlib",
    )

    args = parser.parse_args()

    with open(
        args.program,
        "r",
        encoding="utf-8",
    ) as f:

        source = f.read()

    if args.self_test:

        self_test(
            source
        )

        return

    data = load_input(
        args
    )

    trace = OpticalTrace()
    statistics = OpticalStats()

    variables, machine = run_q(
        source,
        data,
        trace=trace,
        statistics=statistics,
    )

    # If the program produced SHA-256
    # state, print it.

    sha_names = (
        "h0",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "h7",
    )

    if all(
        name in variables
        for name in sha_names
    ):

        print(
            extract_sha256(
                variables
            )
        )

    else:

        print(
            variables
        )

    if args.stats:

        statistics.print()

    if args.trace:

        print()
        print(
            f"Trace events: "
            f"{len(trace.events)}"
        )

        trace.print(
            args.trace_limit
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()