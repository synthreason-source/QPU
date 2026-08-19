"""Reduced-valve optical hardware abstraction layer.

Uses time-multiplexed binary streams instead of unary hole-count arithmetic.
The optical backend is optional: pass a backend object implementing
and_bit(a, b), sum_bits(values), and sample() to OpticalBackend.
Without hardware, the deterministic simulator provides the same API.
"""
from __future__ import annotations
import ast
import argparse
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

MAX_LOOP_ITERATIONS = 100_000


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} expects a nonnegative integer, got {value!r}")
    return value


def _bit(name: str, value: Any) -> int:
    value = _nonnegative_int(name, value)
    if value not in (0, 1):
        raise ValueError(f"{name} expects a binary value, got {value!r}")
    return value


def int_to_bits(value: int, width: Optional[int] = None) -> list[int]:
    value = _nonnegative_int("value", value)
    width = max(1, value.bit_length()) if width is None else width
    if width < 1 or value >= (1 << width):
        raise ValueError("width is too small for value")
    return [(value >> i) & 1 for i in range(width)]


def bits_to_int(bits: Iterable[int]) -> int:
    result = 0
    for i, bit in enumerate(bits):
        result |= _bit("bit", int(bit)) << i
    return result


class SimulatedOptics:
    """Deterministic stand-in for a camera and a two-valve optical path."""
    def and_bit(self, a: int, b: int) -> int:
        return _bit("a", a) & _bit("b", b)

    def sum_bits(self, values: Iterable[int]) -> int:
        return sum(_bit("value", int(v)) for v in values)

    def sample(self, value: int) -> int:
        return int(value)


class OpticalBackend:
    """Adapter for real hardware or a simulator.

    A real backend must provide and_bit(a, b), sum_bits(values), and sample(value).
    The valve_count property describes the number of reusable active valves.
    """
    def __init__(self, driver: Optional[Any] = None, valve_count: int = 2):
        self.driver = driver or SimulatedOptics()
        self.valve_count = valve_count

    def and_bit(self, a: int, b: int) -> int:
        return _bit("AND lhs", self.driver.and_bit(a, b))

    def sum_bits(self, values: Iterable[int]) -> int:
        return self.driver.sum_bits(values)

    def sample(self, value: int) -> int:
        return int(self.driver.sample(value))


@dataclass
class OpticalTrace:
    events: list[tuple[str, tuple[Any, ...], Any]] = field(default_factory=list)

    def record(self, operation: str, inputs: tuple[Any, ...], result: Any) -> Any:
        self.events.append((operation, inputs, result))
        return result

    def print(self) -> None:
        for i, (op, inputs, result) in enumerate(self.events, 1):
            print(f"[{i}] {op}{inputs} -> {result}")


class ReducedValveMachine:
    """Two-valve, time-multiplexed optical arithmetic machine."""
    def __init__(self, backend: Optional[OpticalBackend] = None,
                 width: Optional[int] = None, trace: Optional[OpticalTrace] = None):
        self.backend = backend or OpticalBackend()
        self.width = width
        self.trace = trace or OpticalTrace()

    def _width(self, *values: int) -> int:
        values = [_nonnegative_int("value", v) for v in values]
        width = self.width or max(1, *(v.bit_length() for v in values))
        if any(v >= (1 << width) for v in values):
            raise ValueError("value does not fit configured width")
        return width

    def and_bit(self, a: int, b: int) -> int:
        result = self.backend.and_bit(a, b)
        return self.trace.record("AND", (a, b), result)

    def or_bit(self, a: int, b: int) -> int:
        total = self.backend.sum_bits([a, b])
        result = int(total > 0)
        return self.trace.record("OR", (a, b), result)

    def xor_bit(self, a: int, b: int) -> int:
        total = self.backend.sum_bits([a, b])
        result = total & 1
        return self.trace.record("XOR", (a, b), result)

    def not_bit(self, a: int) -> int:
        result = 1 - _bit("NOT", a)
        return self.trace.record("NOT", (a,), result)

    def _serial_binary(self, a: int, b: int, operation: str) -> int:
        width = self._width(a, b)
        abits, bbits = int_to_bits(a, width), int_to_bits(b, width)
        out = []
        for slot, (ai, bi) in enumerate(zip(abits, bbits)):
            if operation == "AND":
                value = self.backend.and_bit(ai, bi)
            elif operation == "OR":
                value = int(self.backend.sum_bits([ai, bi]) > 0)
            else:
                value = self.backend.sum_bits([ai, bi]) & 1
            out.append(value)
            self.trace.record(f"{operation}[t={slot}]", (ai, bi), value)
        return bits_to_int(out)

    def add(self, a: int, b: int) -> int:
        _nonnegative_int("ADD lhs", a); _nonnegative_int("ADD rhs", b)
        width = max(self._width(a, b) + 1, (a + b).bit_length())
        carry, out = 0, []
        for slot in range(width):
            ai = (a >> slot) & 1
            bi = (b >> slot) & 1
            s = self.backend.sum_bits([ai, bi, carry])
            bit, carry = s & 1, int(s >= 2)
            out.append(bit)
            self.trace.record(f"ADD[t={slot}]", (ai, bi, carry), bit)
        result = bits_to_int(out)
        return self.trace.record("ADD", (a, b), result)

    def subtract(self, a: int, b: int) -> int:
        _nonnegative_int("SUB lhs", a); _nonnegative_int("SUB rhs", b)
        if b > a:
            raise ValueError(f"cannot represent negative result: {a} - {b}")
        width = self._width(a, b)
        borrow, out = 0, []
        for slot in range(width):
            ai, bi = (a >> slot) & 1, (b >> slot) & 1
            difference = ai - bi - borrow
            bit = difference & 1
            borrow = int(difference < 0)
            out.append(bit)
            self.trace.record(f"SUB[t={slot}]", (ai, bi, borrow), bit)
        result = bits_to_int(out)
        return self.trace.record("SUB", (a, b), result)

    def multiply(self, a: int, b: int) -> int:
        _nonnegative_int("MUL lhs", a); _nonnegative_int("MUL rhs", b)
        result, row = 0, a
        shift = 0
        while (b >> shift) > 0:
            if ((b >> shift) & 1):
                result = self.add(result, row)
            self.trace.record(f"MUL[t={shift}]", (a, b), result)
            row <<= 1
            shift += 1
        return self.trace.record("MUL", (a, b), result)

    def compare(self, a: int, b: int) -> int:
        _nonnegative_int("CMP lhs", a); _nonnegative_int("CMP rhs", b)
        if a == b:
            self.trace.record("CMP", (a, b), 0)
            return 0
        result = 1 if a > b else -1
        self.trace.record("CMP", (a, b), result)
        return result

    def eq(self, a: int, b: int) -> int: return int(self.compare(a, b) == 0)
    def ne(self, a: int, b: int) -> int: return int(self.compare(a, b) != 0)
    def lt(self, a: int, b: int) -> int: return int(self.compare(a, b) < 0)
    def le(self, a: int, b: int) -> int: return int(self.compare(a, b) <= 0)
    def gt(self, a: int, b: int) -> int: return int(self.compare(a, b) > 0)
    def ge(self, a: int, b: int) -> int: return int(self.compare(a, b) >= 0)

    def eval(self, expression: str, **variables: int) -> int:
        return _ExpressionEvaluator(self, variables).visit(ast.parse(expression, mode="eval"))

    def run(self, code: str, **variables: int) -> dict[str, int]:
        _StatementExecutor(self, variables).visit(ast.parse(code, mode="exec"))
        return variables


class Bit:
    def __init__(self, value: int, machine: Optional[ReducedValveMachine] = None):
        self.value = _nonnegative_int("Bit", value)
        self.machine = machine or ReducedValveMachine()

    def _other(self, other: Any) -> int:
        return other.value if isinstance(other, Bit) else _nonnegative_int("operand", int(other))

    def _new(self, value: int) -> "Bit": return Bit(value, self.machine)
    def __and__(self, o): return self._new(self.machine._serial_binary(self.value, self._other(o), "AND"))
    def __or__(self, o): return self._new(self.machine._serial_binary(self.value, self._other(o), "OR"))
    def __xor__(self, o): return self._new(self.machine._serial_binary(self.value, self._other(o), "XOR"))
    def __invert__(self): return self._new(self.machine.not_bit(self.value))
    def __add__(self, o): return self._new(self.machine.add(self.value, self._other(o)))
    def __sub__(self, o): return self._new(self.machine.subtract(self.value, self._other(o)))
    def __mul__(self, o): return self._new(self.machine.multiply(self.value, self._other(o)))
    def __int__(self): return self.value
    def __bool__(self): return bool(self.value)
    def __repr__(self): return f"Bit({self.value})"
    def __eq__(self, o): return bool(self.machine.eq(self.value, self._other(o)))
    def __ne__(self, o): return bool(self.machine.ne(self.value, self._other(o)))
    def __lt__(self, o): return bool(self.machine.lt(self.value, self._other(o)))
    def __le__(self, o): return bool(self.machine.le(self.value, self._other(o)))
    def __gt__(self, o): return bool(self.machine.gt(self.value, self._other(o)))
    def __ge__(self, o): return bool(self.machine.ge(self.value, self._other(o)))


class _ExpressionEvaluator(ast.NodeVisitor):
    BIN = {ast.BitAnd: "and_bit", ast.BitOr: "or_bit", ast.BitXor: "xor_bit",
           ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply"}
    BOOL = {ast.And: "and_bit", ast.Or: "or_bit"}
    CMP = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}

    def __init__(self, machine, variables): self.machine, self.variables = machine, variables
    def visit_Expression(self, n): return self.visit(n.body)
    def visit_Name(self, n): return _nonnegative_int(n.id, self.variables[n.id])
    def visit_Constant(self, n): return _nonnegative_int("literal", n.value)
    def visit_BinOp(self, n):
        if type(n.op) not in self.BIN: raise SyntaxError("unsupported operator")
        return getattr(self.machine, self.BIN[type(n.op)])(self.visit(n.left), self.visit(n.right))
    def visit_BoolOp(self, n):
        values = [self.visit(v) for v in n.values]; result = values[0]
        for value in values[1:]: result = getattr(self.machine, self.BOOL[type(n.op)])(result, value)
        return result
    def visit_UnaryOp(self, n):
        if not isinstance(n.op, (ast.Invert, ast.Not)): raise SyntaxError("use ~ or not")
        return self.machine.not_bit(self.visit(n.operand))
    def visit_Compare(self, n):
        left, result = self.visit(n.left), 1
        for op, node in zip(n.ops, n.comparators):
            right = self.visit(node)
            result = self.machine.and_bit(result, getattr(self.machine, self.CMP[type(op)])(left, right))
            left = right
        return result
    def generic_visit(self, n): raise SyntaxError(f"unsupported syntax: {type(n).__name__}")


class _Break(Exception): pass
class _Continue(Exception): pass

class _StatementExecutor(ast.NodeVisitor):
    def __init__(self, machine, variables): self.machine, self.variables = machine, variables
    def evaluate(self, node): return _ExpressionEvaluator(self.machine, self.variables).visit(node)
    def visit_Module(self, n):
        for statement in n.body: self.visit(statement)
    def visit_Pass(self, n): pass
    def visit_Break(self, n): raise _Break()
    def visit_Continue(self, n): raise _Continue()
    def visit_Assign(self, n):
        if len(n.targets) != 1 or not isinstance(n.targets[0], ast.Name): raise SyntaxError("simple assignments only")
        self.variables[n.targets[0].id] = self.evaluate(n.value)
    def visit_AugAssign(self, n):
        if not isinstance(n.target, ast.Name): raise SyntaxError("simple augmented assignments only")
        op = _ExpressionEvaluator.BIN.get(type(n.op))
        if op is None: raise SyntaxError("unsupported augmented operator")
        self.variables[n.target.id] = getattr(self.machine, op)(self.variables[n.target.id], self.evaluate(n.value))
    def visit_Expr(self, n):
        if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "print":
            print(" ".join(str(self.evaluate(arg)) for arg in n.value.args)); return
        self.evaluate(n.value)
    def visit_If(self, n):
        for test, body in [(n.test, n.body)]:
            if self.evaluate(test):
                for s in body: self.visit(s)
                return
        for s in n.orelse: self.visit(s)
    def visit_While(self, n):
        count = 0
        while self.evaluate(n.test):
            count += 1
            if count > MAX_LOOP_ITERATIONS: raise RuntimeError("loop limit exceeded")
            try:
                for s in n.body: self.visit(s)
            except _Break: break
            except _Continue: continue
    def visit_For(self, n):
        if not isinstance(n.target, ast.Name): raise SyntaxError("simple for targets only")
        if not (isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range"):
            raise SyntaxError("for loops require range(...)")
        bounds = [self.evaluate(a) for a in n.iter.args]
        for value in range(*bounds):
            self.variables[n.target.id] = value
            try:
                for s in n.body: self.visit(s)
            except _Break: break
            except _Continue: continue
    def generic_visit(self, n): raise SyntaxError(f"unsupported statement: {type(n).__name__}")


def optical_eval(expression: str, **variables: int) -> tuple[int, OpticalTrace]:
    machine = ReducedValveMachine()
    result = machine.eval(expression, **variables)
    return result, machine.trace


def optical_run(code: str, **variables: int) -> tuple[dict[str, int], OpticalTrace]:
    machine = ReducedValveMachine()
    result = machine.run(code, **variables)
    return result, machine.trace


def demo() -> None:
    machine = ReducedValveMachine(width=8)
    a, b = Bit(13, machine), Bit(6, machine)
    print("13 + 6 =", a + b)
    print("13 * 6 =", a * b)
    print("13 & 6 =", a & b)
    print("13 | 6 =", a | b)
    print("13 ^ 6 =", a ^ b)
    print("exposures/events:", len(machine.trace.events))
    machine.trace.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduced-valve optical computing interpreter")
    parser.add_argument("--expr", help="evaluate one expression")
    parser.add_argument("--program", help="execute a program file")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("values", nargs="*", help="initial values such as a=13")
    args = parser.parse_args()
    values = {k: int(v) for item in args.values for k, v in [item.split("=", 1)]}
    if args.demo: demo()
    elif args.expr:
        result, trace = optical_eval(args.expr, **values)
        print(result); trace.print()
    elif args.program:
        variables, trace = optical_run(open(args.program, encoding="utf-8").read(), **values)
        print(variables); trace.print()
    else:
        parser.print_help()
