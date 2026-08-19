#!/usr/bin/env python3
"""Reduced-valve optical computing simulator with runtime statistics."""
from __future__ import annotations
import argparse, ast
from dataclasses import dataclass, field
from typing import Any

MAX_LOOP_ITERATIONS = 100_000

def nonnegative(value: Any, name: str = "value") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return value

def binary(value: Any, name: str = "bit") -> int:
    value = nonnegative(value, name)
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value

def to_bits(value: int, width: int | None = None) -> list[int]:
    value = nonnegative(value)
    width = max(1, value.bit_length()) if width is None else width
    if width < 1 or value >= (1 << width):
        raise ValueError("value does not fit configured width")
    return [(value >> index) & 1 for index in range(width)]

def from_bits(values: list[int]) -> int:
    return sum(binary(value) << index for index, value in enumerate(values))

class SimulatedOptics:
    def and_bit(self, a: int, b: int) -> int:
        return binary(a, "a") & binary(b, "b")
    def sum_bits(self, values: list[int]) -> int:
        return sum(binary(value) for value in values)
    def sample(self, value: int) -> int:
        return int(value)

class OpticalBackend:
    def __init__(self, driver: Any | None = None, valve_count: int = 2):
        self.driver = driver or SimulatedOptics()
        self.valve_count = valve_count
    def and_bit(self, a: int, b: int) -> int:
        return binary(self.driver.and_bit(a, b), "AND result")
    def sum_bits(self, values: list[int]) -> int:
        return int(self.driver.sum_bits(values))
    def sample(self, value: int) -> int:
        return int(self.driver.sample(value))

@dataclass
class OpticalTrace:
    events: list[tuple[str, tuple[Any, ...], Any]] = field(default_factory=list)
    def record(self, operation: str, inputs: tuple[Any, ...], result: Any) -> Any:
        self.events.append((operation, inputs, result))
        return result
    def print(self) -> None:
        for index, (operation, inputs, result) in enumerate(self.events, 1):
            print(f"[{index}] {operation}{inputs} -> {result}")
    def stats(self, valve_count: int = 2, width: int | None = None) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for operation, _, _ in self.events:
            base = operation.split("[", 1)[0]
            counts[base] = counts.get(base, 0) + 1
        return {
            "events": len(self.events),
            "bit_slot_events": sum("[t=" in operation for operation, _, _ in self.events),
            "operation_counts": dict(sorted(counts.items())),
            "reusable_valves": valve_count,
            "configured_width": width,
            "encoding": "binary time multiplexing",
        }
    def print_stats(self, valve_count: int = 2, width: int | None = None) -> None:
        stats = self.stats(valve_count, width)
        print("Optical machine statistics")
        print("--------------------------")
        print(f"Reusable valves: {stats['reusable_valves']}")
        print(f"Encoding: {stats['encoding']}")
        if stats["configured_width"] is not None:
            print(f"Configured width: {stats['configured_width']}")
        print(f"Total trace events: {stats['events']}")
        print(f"Bit-slot events: {stats['bit_slot_events']}")
        print("Operation counts:")
        for operation, count in stats["operation_counts"].items():
            print(f"  {operation}: {count}")

class ReducedValveMachine:
    def __init__(self, backend: OpticalBackend | None = None, width: int | None = None, trace: OpticalTrace | None = None):
        self.backend = backend or OpticalBackend()
        self.width = width
        self.trace = trace or OpticalTrace()
    def width_for(self, *values: int) -> int:
        values = [nonnegative(value) for value in values]
        width = self.width or max(1, *(value.bit_length() for value in values))
        if any(value >= (1 << width) for value in values):
            raise ValueError("value does not fit configured width")
        return width
    def and_bit(self, a: int, b: int) -> int:
        result = self.backend.and_bit(a, b)
        return self.trace.record("AND", (a, b), result)
    def or_bit(self, a: int, b: int) -> int:
        result = int(self.backend.sum_bits([a, b]) > 0)
        return self.trace.record("OR", (a, b), result)
    def xor_bit(self, a: int, b: int) -> int:
        result = self.backend.sum_bits([a, b]) & 1
        return self.trace.record("XOR", (a, b), result)
    def not_bit(self, value: int) -> int:
        result = 1 - binary(value)
        return self.trace.record("NOT", (value,), result)
    def serial_logic(self, a: int, b: int, operation: str) -> int:
        width = self.width_for(a, b)
        output = []
        for slot, (ai, bi) in enumerate(zip(to_bits(a, width), to_bits(b, width))):
            if operation == "AND": result = self.and_bit(ai, bi)
            elif operation == "OR": result = self.or_bit(ai, bi)
            elif operation == "XOR": result = self.xor_bit(ai, bi)
            else: raise ValueError("unknown logic operation")
            self.trace.record(f"{operation}[t={slot}]", (ai, bi), result)
            output.append(result)
        return from_bits(output)
    def add(self, a: int, b: int) -> int:
        nonnegative(a, "ADD lhs"); nonnegative(b, "ADD rhs")
        carry, output = 0, []
        for slot in range(max(self.width_for(a, b) + 1, (a + b).bit_length())):
            ai, bi = (a >> slot) & 1, (b >> slot) & 1
            total = self.backend.sum_bits([ai, bi, carry])
            result, carry = total & 1, int(total >= 2)
            self.trace.record(f"ADD[t={slot}]", (ai, bi, carry), result)
            output.append(result)
        return self.trace.record("ADD", (a, b), from_bits(output))
    def subtract(self, a: int, b: int) -> int:
        nonnegative(a, "SUB lhs"); nonnegative(b, "SUB rhs")
        if b > a: raise ValueError(f"cannot represent negative result: {a} - {b}")
        borrow, output = 0, []
        for slot in range(self.width_for(a, b)):
            ai, bi = (a >> slot) & 1, (b >> slot) & 1
            difference = ai - bi - borrow
            result, borrow = difference & 1, int(difference < 0)
            self.trace.record(f"SUB[t={slot}]", (ai, bi, borrow), result)
            output.append(result)
        return self.trace.record("SUB", (a, b), from_bits(output))
    def multiply(self, a: int, b: int) -> int:
        nonnegative(a, "MUL lhs"); nonnegative(b, "MUL rhs")
        result = 0
        for slot in range(b.bit_length()):
            if (b >> slot) & 1: result = self.add(result, a << slot)
            self.trace.record(f"MUL[t={slot}]", (a, b), result)
        return self.trace.record("MUL", (a, b), result)
    def compare(self, a: int, b: int) -> int:
        result = (a > b) - (a < b)
        return self.trace.record("CMP", (a, b), result)
    def eq(self, a, b): return int(self.compare(a, b) == 0)
    def ne(self, a, b): return int(self.compare(a, b) != 0)
    def lt(self, a, b): return int(self.compare(a, b) < 0)
    def le(self, a, b): return int(self.compare(a, b) <= 0)
    def gt(self, a, b): return int(self.compare(a, b) > 0)
    def ge(self, a, b): return int(self.compare(a, b) >= 0)
    def stats(self): return self.trace.stats(self.backend.valve_count, self.width)
    def print_stats(self): self.trace.print_stats(self.backend.valve_count, self.width)
    def eval(self, expression: str, **variables: int) -> int: return ExpressionEvaluator(self, variables).visit(ast.parse(expression, mode="eval"))
    def run(self, code: str, **variables: int) -> dict[str, int]: StatementExecutor(self, variables).visit(ast.parse(code, mode="exec")); return variables

class Bit:
    def __init__(self, value: int, machine: ReducedValveMachine | None = None): self.value, self.machine = nonnegative(value, "Bit"), machine or ReducedValveMachine()
    def other(self, value): return value.value if isinstance(value, Bit) else nonnegative(int(value), "operand")
    def new(self, value): return Bit(value, self.machine)
    def __and__(self, value): return self.new(self.machine.serial_logic(self.value, self.other(value), "AND"))
    def __or__(self, value): return self.new(self.machine.serial_logic(self.value, self.other(value), "OR"))
    def __xor__(self, value): return self.new(self.machine.serial_logic(self.value, self.other(value), "XOR"))
    def __invert__(self): return self.new(self.machine.not_bit(self.value))
    def __add__(self, value): return self.new(self.machine.add(self.value, self.other(value)))
    def __sub__(self, value): return self.new(self.machine.subtract(self.value, self.other(value)))
    def __mul__(self, value): return self.new(self.machine.multiply(self.value, self.other(value)))
    def __int__(self): return self.value
    def __bool__(self): return bool(self.value)
    def __repr__(self): return f"Bit({self.value})"
    def __eq__(self, value): return bool(self.machine.eq(self.value, self.other(value)))
    def __ne__(self, value): return bool(self.machine.ne(self.value, self.other(value)))
    def __lt__(self, value): return bool(self.machine.lt(self.value, self.other(value)))
    def __le__(self, value): return bool(self.machine.le(self.value, self.other(value)))
    def __gt__(self, value): return bool(self.machine.gt(self.value, self.other(value)))
    def __ge__(self, value): return bool(self.machine.ge(self.value, self.other(value)))

class ExpressionEvaluator(ast.NodeVisitor):
    BIN = {ast.BitAnd: "serial_logic", ast.BitOr: "serial_logic", ast.BitXor: "serial_logic", ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply"}
    CMP = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}
    def __init__(self, machine, variables): self.machine, self.variables = machine, variables
    def visit_Expression(self, node): return self.visit(node.body)
    def visit_Name(self, node): return nonnegative(self.variables[node.id], node.id)
    def visit_Constant(self, node): return nonnegative(node.value, "literal")
    def visit_BinOp(self, node):
        if type(node.op) in (ast.BitAnd, ast.BitOr, ast.BitXor): return getattr(self.machine, self.BIN[type(node.op)])(self.visit(node.left), self.visit(node.right), {ast.BitAnd:"AND",ast.BitOr:"OR",ast.BitXor:"XOR"}[type(node.op)])
        return getattr(self.machine, self.BIN[type(node.op)])(self.visit(node.left), self.visit(node.right))
    def visit_BoolOp(self, node):
        result = self.visit(node.values[0]); method = self.machine.and_bit if isinstance(node.op, ast.And) else self.machine.or_bit
        for value in node.values[1:]: result = method(result, self.visit(value))
        return result
    def visit_UnaryOp(self, node): return self.machine.not_bit(self.visit(node.operand))
    def visit_Compare(self, node):
        left, result = self.visit(node.left), 1
        for op, item in zip(node.ops, node.comparators):
            right = self.visit(item); result = self.machine.and_bit(result, getattr(self.machine, self.CMP[type(op)])(left, right)); left = right
        return result
    def generic_visit(self, node): raise SyntaxError(f"unsupported syntax: {type(node).__name__}")

class BreakSignal(Exception): pass
class ContinueSignal(Exception): pass
class StatementExecutor(ast.NodeVisitor):
    def __init__(self, machine, variables): self.machine, self.variables = machine, variables
    def evaluate(self, node): return ExpressionEvaluator(self.machine, self.variables).visit(node)
    def visit_Module(self, node):
        for statement in node.body: self.visit(statement)
    def visit_Assign(self, node): self.variables[node.targets[0].id] = self.evaluate(node.value)
    def visit_AugAssign(self, node): self.variables[node.target.id] = self.machine.add(self.variables[node.target.id], self.evaluate(node.value)) if isinstance(node.op, ast.Add) else self.machine.subtract(self.variables[node.target.id], self.evaluate(node.value))
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "print": print(" ".join(str(self.evaluate(arg)) for arg in node.value.args))
        else: self.evaluate(node.value)
    def visit_If(self, node):
        for statement in node.body if self.evaluate(node.test) else node.orelse: self.visit(statement)
    def visit_While(self, node):
        count = 0
        while self.evaluate(node.test):
            count += 1
            if count > MAX_LOOP_ITERATIONS: raise RuntimeError("loop limit exceeded")
            try:
                for statement in node.body: self.visit(statement)
            except BreakSignal: break
            except ContinueSignal: continue
    def visit_For(self, node):
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range": raise SyntaxError("for loops require range(...)")
        for value in range(*[self.evaluate(arg) for arg in node.iter.args]):
            self.variables[node.target.id] = value
            try:
                for statement in node.body: self.visit(statement)
            except BreakSignal: break
            except ContinueSignal: continue
    def visit_Break(self, node): raise BreakSignal()
    def visit_Continue(self, node): raise ContinueSignal()
    def visit_Pass(self, node): pass
    def generic_visit(self, node): raise SyntaxError(f"unsupported statement: {type(node).__name__}")

def demo():
    machine = ReducedValveMachine(width=8); a, b = Bit(13, machine), Bit(6, machine)
    print("13 + 6 =", a + b); print("13 * 6 =", a * b); print("13 & 6 =", a & b); machine.print_stats(); machine.trace.print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduced-valve optical machine")
    parser.add_argument("--expr"); parser.add_argument("--program"); parser.add_argument("--stats", action="store_true"); parser.add_argument("--trace", action="store_true"); parser.add_argument("--demo", action="store_true"); parser.add_argument("values", nargs="*", help="name=value")
    args = parser.parse_args(); values = {key: int(value) for item in args.values for key, value in [item.split("=", 1)]}
    if args.demo: demo()
    elif args.expr:
        machine = ReducedValveMachine(); print(machine.eval(args.expr, **values)); machine.print_stats() if args.stats else None; machine.trace.print() if args.trace else None
    elif args.program:
        machine = ReducedValveMachine(); print(machine.run(open(args.program, encoding="utf-8").read(), **values)); machine.print_stats() if args.stats else None; machine.trace.print() if args.trace else None
    else: parser.print_help()
