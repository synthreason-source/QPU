
**An Optical Hardware Abstraction Layer for Programmable Boolean and Arithmetic Computation**

## Abstract

Optical computing offers a promising route toward highly parallel and physically direct information processing. However, programming optical hardware remains difficult when the underlying device exposes only a small set of physical primitives. This paper presents an optical hardware abstraction layer that transforms low-level optical operations into a programmable interface supporting Boolean algebra, arithmetic, comparisons, expression evaluation, and structured control flow.

The abstraction is built around two fundamental operations: optical binary multiplication, interpreted as an AND operation, and optical summation over an aperture containing configurable holes. Higher-level operations are constructed from these primitives without bypassing the optical execution model. Boolean OR and XOR are implemented by camera-based summation followed by thresholding or parity reduction. Nonnegative integer arithmetic uses unary hole-count encoding, with addition represented by the aggregation of open holes, subtraction by selective aperture closure, and multiplication by repeated optical addition. Comparisons reuse the subtractive mechanism to infer ordering.

Three programming interfaces are provided: an operator-overloaded `Bit` object, an expression evaluator called `ooeval`, and a restricted program executor called `oorun`. These interfaces allow users to write expressions and small programs using conventional Python-like syntax while maintaining a continuous trace of optical operations. The approach demonstrates how a narrowly defined optical instruction set can be exposed through a higher-level programming model. Nevertheless, the unary encoding and repeated-addition strategy introduce substantial scaling limitations, motivating future work in spatial parallelism, encoded arithmetic, optical thresholding, and hardware-aware compilation.

**Keywords:** optical computing, hardware abstraction layer, photonic logic, unary encoding, optical arithmetic, camera readout, domain-specific language, quantum processing unit abstraction

## 1. Introduction

Conventional computer architectures expose arithmetic and logical operations through electronic circuits, instruction sets, and memory hierarchies. Optical computing follows a different model: information is encoded in properties of light, manipulated through optical components, and measured by photodetectors or cameras. The physical operations can be highly parallel, but they are often less convenient to program than conventional digital processors.

A practical optical processor may expose only a small number of native operations. In the system considered here, the principal primitives are binary multiplication and optical summation. Binary multiplication is used to implement Boolean conjunction, while summation is performed by placing multiple values on an optical plane and reading the resulting camera intensity. The challenge is therefore not merely to define individual optical operations, but to create an abstraction that permits meaningful computation while preserving the physical execution semantics.

This paper introduces a software abstraction layer for such a system. The layer provides three levels of access:

1. A value object whose operators invoke optical primitives.
2. An expression evaluator that interprets a restricted Python-like syntax.
3. A program executor supporting assignments, conditional statements, loops, and output.

The design goal is to make optical execution resemble ordinary programming without pretending that the underlying computation is electronic. Every supported operation is reduced to the available optical backend, and each execution is recorded in a trace that exposes the sequence of optical operations.

The contribution of this work is therefore architectural rather than purely algorithmic. It demonstrates how a minimal optical instruction set can be transformed into a domain-specific programming environment.

## 2. Optical Computational Model

### 2.1 Native primitives

The abstraction layer assumes an optical backend containing operations equivalent to:

- `binary_multiply(x, y)`: performs a binary optical multiplication and camera readout.
- `binary_add_many(values)`: creates an optical plane containing multiple values and measures the resulting signal.
- `make_binary_plane_and_holes(...)`: creates an optical representation with configurable open and closed regions.
- `apply_holes(...)`: modifies the active optical aperture.
- `camera_exposure(...)`: performs optical propagation and returns a measured field.

The software layer does not directly replace these operations with ordinary arithmetic. Instead, it treats them as the physical instruction set from which higher-level functions are composed.[^1]

### 2.2 Binary representation

Binary values are represented as integers restricted to $0$ and $1$. The optical AND operation is defined directly as

$$
x \land y = xy.
$$

The implementation therefore maps Boolean conjunction to the native binary multiplication primitive:

$$
\operatorname{opt\_and}(x,y)
=
\operatorname{binary\_multiply}(x,y).
$$

For summation-based operations, the camera measures a total signal:

$$
S(x_1,\ldots,x_n)
=
\sum_{i=1}^{n} x_i.
$$

This sum can then be transformed into another Boolean operation.

### 2.3 Boolean operations

Optical OR is obtained by thresholding the measured sum:

$$
x \lor y =
\begin{cases}
1, & x+y \geq 1,\\
0, & x+y=0.
\end{cases}
$$

Optical XOR is obtained by reducing the sum modulo two:

$$
x \oplus y = (x+y)\bmod 2.
$$

The complement operation is implemented as

$$
\lnot x = 1-x,
$$

for $x\in\{0,1\}$. Although this operation does not require a camera exposure in the current implementation, it remains part of the optical programming abstraction because it describes the complementary aperture state.[^1]

The resulting Boolean algebra is sufficiently expressive to represent conventional logical expressions. For example,

$$
(a\land b)\lor(\lnot c\land d)
$$

can be evaluated through a sequence of optical AND, NOT, and OR operations.

## 3. Optical Arithmetic

### 3.1 Unary hole-count encoding

The arithmetic layer represents a nonnegative integer $n$ using $n$ open optical holes. This is a unary representation:

$$
n \longleftrightarrow
\underbrace{1+1+\cdots+1}_{n\text{ open holes}}.
$$

The representation is physically intuitive because the measured intensity corresponds directly to the number of transmitting elements. However, its storage and execution cost grows linearly with the represented value.

### 3.2 Addition

Addition combines the open holes associated with two operands:

$$
a+b
=
\sum_{i=1}^{a}1+\sum_{j=1}^{b}1.
$$

The implementation constructs a list containing $a+b$ unit-valued entries and submits it to the optical summation primitive. The camera readout then returns the total count.

This approach provides a direct physical interpretation of addition: the optical system aggregates the two populations of open holes in a single measurement.

### 3.3 Subtraction

Subtraction begins with an aperture containing $a$ open holes and closes $b$ of them:

$$
a-b =
\left|\{\text{open holes remaining after closing }b\text{ holes}\}\right|,
$$

provided $b\leq a$. If $b>a$, the implementation rejects the operation because the aperture cannot represent a negative hole count.

The subtraction routine therefore defines arithmetic over the nonnegative integers:

$$
\operatorname{optical\_subtract}:
\mathbb{N}_0\times\mathbb{N}_0
\rightarrow \mathbb{N}_0,
$$

with the domain restriction $a\geq b$.[^1]

### 3.4 Multiplication

For general nonnegative integers, multiplication is implemented through repeated addition:

$$
a\times b
=
\underbrace{a+a+\cdots+a}_{b\text{ times}}.
$$

The computational cost is therefore proportional to $b$, in addition to the cost of representing the intermediate unary values. When both operands are binary values, multiplication is delegated to the native optical AND operation because

$$
x y = x\land y
\quad\text{for}\quad x,y\in\{0,1\}.
$$

This distinction is important: the same abstraction supports both native binary logic and generalized nonnegative arithmetic, but the two modes have different performance characteristics.

## 4. Comparison Operations

The comparison layer reuses the subtractive mechanism. To compare $a$ and $b$, the system first attempts to evaluate $a-b$.

- If $a-b=0$, then $a=b$.
- If $a-b>0$, then $a>b$.
- If the subtraction fails because $b>a$, the system evaluates $b-a$, establishing that $a<b$.

The comparison function therefore returns

$$
\operatorname{cmp}(a,b)=
\begin{cases}
-1, & a<b,\\
0, & a=b,\\
1, & a>b.
\end{cases}
$$

Equality and ordering predicates are then derived from this three-way comparison.

This construction is conceptually simple, but it exposes an important property of the abstraction: control decisions are not primitive hardware operations. They are inferred from optical measurements and aperture constraints.

## 5. Programming Interfaces

### 5.1 The `Bit` object

The `Bit` class provides operator overloading so that ordinary Python syntax can invoke optical computation. Boolean operators include:

- `&` for AND.
- `|` for OR.
- `^` for XOR.
- `~` for NOT.

Arithmetic operators include:

- `+` for optical addition.
- `-` for optical subtraction.
- `*` for optical multiplication.

Comparison operators are also supported. Each operation produces a new `Bit` object containing the result and an accumulated trace of operations. This trace records the operation name, operands, and result, enabling post-execution inspection of the optical computation.[^1]

For example:

```python
a = Bit(1)
b = Bit(0)
c = Bit(1)

result = (a & b) | c
```

Conceptually, the expression expands into an optical AND exposure followed by an optical OR exposure.

### 5.2 Expression evaluation

The `ooeval` interface parses an expression into an abstract syntax tree. A restricted visitor evaluates supported syntax nodes and maps each operator to its optical implementation.

Supported expressions include:

```python
(a & b) | ~c
a + b * c
(a + b) - c
a == b
a < b < c
```

The evaluator avoids unrestricted Python execution. Unsupported syntax raises an error, which is preferable for a hardware interface because it prevents accidental execution of operations that have no optical interpretation.

Expression evaluation also supports trace collection. The trace can be interpreted as a software-level optical instruction stream.

### 5.3 Program execution

The `oorun` interface extends expression evaluation to a small imperative language. Its supported constructs include:

- Simple assignments.
- Augmented assignments.
- `if`, `elif`, and `else`.
- `while` loops.
- `for` loops over `range(...)` or literal lists.
- `break` and `continue`.
- `print(...)`.

For example:

```python
total = 0
for i in range(5):
    total = total + i

print(total)
```

The loop itself is executed by the host processor, but every arithmetic operation used to update `total` is routed through the optical arithmetic layer. Similarly, loop conditions and conditional predicates are evaluated using optical comparisons. This creates a hybrid execution model in which control flow is software-managed while data operations are delegated to the optical backend.

## 6. Execution Tracing

A central feature of the abstraction is continuous trace collection. Each optical operation contributes a record of the form

$$
(\text{operation},\text{inputs},\text{result}).
$$

For example, evaluating

```python
a + b * c
```

with $a=2$, $b=3$, and $c=4$ produces a conceptual trace containing:

1. Multiplication of $3$ and $4$.
2. Addition of $2$ and the multiplication result.

The trace serves several purposes:

- Debugging optical programs.
- Verifying that high-level expressions are correctly lowered.
- Measuring the number of exposures.
- Studying the relationship between source syntax and physical operations.
- Providing an intermediate representation for future optimization.

In a research system, trace data could also be extended with physical metadata such as exposure duration, camera frame number, optical power, detected intensity, noise estimates, and aperture dimensions.

## 7. Complexity and Scalability

The principal limitation of the current system is unary encoding. If an integer $n$ is represented by $n$ open holes, then the physical resource requirement is

$$
M(n)=O(n).
$$

For addition, the number of aperture elements is proportional to $a+b$. For multiplication implemented by repeated addition, the number of optical additions is proportional to $b$, while the total amount of manipulated unary data may grow with $ab$.

A simplified cost model is:


| Operation | Optical strategy | Approximate exposure behavior |
| :-- | :-- | --: |
| AND | Native binary multiplication | $O(1)$ exposure |
| OR | Summation and threshold | $O(1)$ exposure |
| XOR | Summation and parity reduction | $O(1)$ exposure |
| NOT | Complementary state | $O(1)$ logical operation |
| Addition | Unary hole summation | $O(1)$ readout, $O(a+b)$ elements |
| Subtraction | Hole closure and readout | $O(1)$ readout, $O(a)$ elements |
| Multiplication | Repeated optical addition | $O(b)$ additions |
| Comparison | One or two subtractions | Up to two subtraction operations |

The distinction between exposure count and spatial resource count is important. An operation may require only one camera readout while still requiring a large aperture or many individually controlled holes.

The current design also assumes ideal or nearly ideal measurement behavior. In a physical implementation, optical noise, detector saturation, nonuniform illumination, diffraction, crosstalk, imperfect hole closure, and calibration drift could cause the measured result to deviate from the intended integer. Robust decoding would therefore require tolerance intervals rather than exact integer rounding.

## 8. Security and Safety of the Interpreter

Because `ooeval` and `oorun` process user-provided code, restricting the accepted syntax is essential. The AST-based design is safer than calling Python’s unrestricted `eval` or `exec`, because only explicitly handled node types are accepted.

The current restrictions exclude:

- Function calls other than `print`.
- Attribute access.
- Imports.
- Arbitrary indexing.
- Object construction.
- Negative literals.
- Floating-point values.
- General Python expressions outside the supported operator set.

The loop iteration limit is another important safeguard. A maximum of 100,000 iterations prevents an accidentally infinite `while` loop from running indefinitely.[^1]

For deployment on physical optical equipment, additional safeguards would be needed:

- Maximum aperture size.
- Maximum optical power.
- Maximum exposure duration.
- Camera saturation detection.
- Hardware emergency shutdown.
- Resource quotas per program.
- Validation of all initial variables.
- Timeouts for optical and camera operations.


## 9. Limitations

Several limitations constrain the current abstraction.

First, unary representation is inefficient for large integers. Binary, Gray-coded, residue, or spatially distributed encodings could reduce the number of required optical elements.

Second, multiplication is not intrinsically optical for general integers in the current design. It is implemented through repeated addition, which limits scalability.

Third, comparisons depend on subtraction failure as an ordering signal. A real physical device may not naturally “raise an exception” when a requested subtraction is invalid. A hardware implementation would need an underflow flag, saturation detector, or signed encoding.

Fourth, control flow remains host-controlled. The optical system evaluates data-dependent conditions, but the host processor performs branching and iteration. The architecture is therefore better described as a hybrid optical coprocessor interface than as a fully optical general-purpose processor.

Fifth, the model conflates abstract hole counts with measured optical intensity. This is acceptable for simulation or calibrated experimentation, but a physical system would require a formal measurement model:

$$
I_{\text{measured}}
=
g(I_{\text{ideal}})
+\eta,
$$

where $g$ represents detector response and $\eta$ represents noise and systematic error.

Finally, the current trace records logical operations but not all physical state. A complete experimental trace should include the optical plane, hole mask, camera field, calibration values, and decoding confidence.

## 10. Future Work

Future development could proceed in several directions.

### 10.1 Encoded arithmetic

Replacing unary encoding with binary spatial encoding would substantially reduce resource requirements. A binary integer could be represented across multiple optical modes:

$$
n=\sum_{k=0}^{m-1} b_k2^k,
\quad b_k\in\{0,1\}.
$$

This would require optical carry propagation or parallel carry-lookahead mechanisms, but it would provide logarithmic spatial scaling.

### 10.2 Parallel multiplication

Repeated addition could be replaced with optical convolution, interferometric multiplication, or tensor-style parallel operations. Spatial light modulators and camera arrays may permit many partial products to be generated simultaneously.

### 10.3 Hardware-aware compilation

The AST evaluator could be extended into a compiler that:

- Simplifies constant expressions.
- Merges compatible optical exposures.
- Reuses intermediate results.
- Estimates aperture and exposure costs.
- Selects between native Boolean operations and arithmetic encodings.
- Produces a scheduled optical instruction stream.


### 10.4 Noise-aware decoding

Rather than applying exact thresholding, the system could use calibrated statistical decoding. A measured camera signal $I$ could be assigned to the most likely logical value:

$$
\hat{x}
=
\arg\max_{x\in\mathcal{X}} P(x\mid I).
$$

This would allow the abstraction to operate reliably under realistic detector noise.

### 10.5 Differentiable optical programming

Because optical systems can implement linear transformations naturally, the abstraction could be extended toward differentiable programming. Optical traces might then serve as computational graphs for gradient-based optimization of masks, phase profiles, or photonic circuit parameters.

## 11. Conclusion

This paper presented a programmable abstraction layer for an optical computational backend with a small set of native primitives. By mapping binary multiplication to AND, camera-based summation to OR and XOR, aperture modification to subtraction, and repeated optical addition to multiplication, the system exposes a conventional programming interface while preserving the underlying optical execution model.

The `Bit` object provides operator-level access, `ooeval` supports restricted expression evaluation, and `oorun` enables small structured programs. Continuous operation tracing gives the system a useful intermediate representation for debugging, verification, cost analysis, and future compilation.[^1]

The architecture demonstrates that even a minimal optical instruction set can support a meaningful programming environment. Its principal limitation is the use of unary hole-count arithmetic, which imposes linear spatial cost and repeated-exposure overhead. Nevertheless, the abstraction provides a foundation for more advanced optical compilers, noise-aware execution models, spatially parallel arithmetic, and hybrid optical-electronic computing systems.

Author - George Wagenknecht - 2026
