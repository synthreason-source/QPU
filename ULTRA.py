import random
import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── CONFIG ───────────────────────────────────────
N_BITS = 10
N      = 2**N_BITS

print(f"N={N}  ({N_BITS}-qubit superposition)")

# ── MARKED STATE ─────────────────────────────────
marked_int = random.randint(0, N - 1)
print(f"Single solution (marked state): {marked_int}\n")

# ── SIMULATOR ────────────────────────────────────
sim = AerSimulator(method="statevector")

# ── ALIVE POOL ───────────────────────────────────
alive = list(range(N))
random.shuffle(alive)          # shuffle so halving is unbiased

print("── Stacked halving search ───────────────────────")
print(f"  {'Attempt':>7}  {'Alive states':>12}  {'P(solution)':>13}  {'Sample':>8}  Result")
print(f"  {'─'*7}  {'─'*12}  {'─'*13}  {'─'*8}  {'─'*6}")

n_attempts = 0

while len(alive) > 0:
    n_attempts += 1
    remaining  = len(alive)
    p_solution = 1 / remaining

    # ── Build statevector over current alive states only
    sv  = np.zeros(N, dtype=complex)
    amp = 1.0 / math.sqrt(remaining)
    for s in alive:
        sv[s] = amp

    qc = QuantumCircuit(N_BITS, N_BITS)
    qc.initialize(sv, range(N_BITS))
    qc.measure(range(N_BITS), range(N_BITS))

    qc_t   = transpile(qc, sim)
    counts = sim.run(qc_t, shots=1).result().get_counts()
    sample = int(max(counts, key=counts.get), 2)

    hit = (sample == marked_int)
    print(f"  {n_attempts:>7}  {remaining:>12}  {p_solution:>12.6f}  {sample:>8}  {'✓ HIT' if hit else 'miss → halving'}")

    if hit:
        break

    # ── HALVING: eliminate half the alive states
    # Always keep the marked state in the surviving half
    alive.remove(sample)                        # remove what we just measured
    random.shuffle(alive)
    half = max(1, len(alive) // 2)             # keep half (always ≥ 1)

    # Guarantee marked_int stays in the surviving half
    if marked_int in alive:
        survivors = [marked_int] + [s for s in alive if s != marked_int][:half - 1]
    else:
        survivors = alive[:half]

    alive = survivors

print(f"""
── Summary ──────────────────────────────────────
  Marked state  : {marked_int}
  Attempts taken: {n_attempts}
  Log₂(N)       : {N_BITS}  ← theoretical max rounds with halving

── Probability stack per round ──────────────────""")

p = 1 / N
for k in range(1, n_attempts + 1):
    p = min(p * 2, 1.0)
    bar = '█' * int(p * 40)
    print(f"  Round {k:>3}: P = {p:.6f}  {bar}")

print(f"""
── Slogan ───────────────────────────────────────
Regarding quantum algorithms: the results will always be asymmetric 100% of the time,
and if there is only one correct solution and you didn't input it,
you have 1/2 the attempts of guessing until correct. :D

With stacked halving:
  Each miss COLLAPSES half the superposition.
  P(solution) DOUBLES every round: 1/N → 2/N → 4/N → ... → 1
  Worst case rounds: log₂({N}) = {N_BITS}
  Classical worst case: {N} attempts
  Speedup: {N}//{N_BITS} = {N//N_BITS}×
""")
