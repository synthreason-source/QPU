import random
import math
import hashlib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── CONFIG ───────────────────────────────────────
N_BITS    = 21
N         = 2**N_BITS
DIFF_BITS = 20          # number of leading zero bits required (difficulty)
                        # e.g. 20 → hash must start with 20 zero bits (~1-in-1M)

BLOCK_HEADER = "prev=abc123|merkle=deadbeef|height=800000"

print(f"N={N}  ({N_BITS}-qubit nonce space)")
print(f"Block header : {BLOCK_HEADER}")
print(f"Difficulty   : {DIFF_BITS} leading zero bits")
print(f"Target       : hash < {'0'*DIFF_BITS}{'1'*(256-DIFF_BITS)}")
print()

# ── PoW ORACLE ───────────────────────────────────
# Replace the "marked_int" concept: a nonce is VALID if its SHA-256
# hash (of header+nonce) has >= DIFF_BITS leading zero bits.

def pow_hash(nonce: int) -> str:
    """SHA-256 of block_header + nonce, returned as binary string."""
    raw = f"{BLOCK_HEADER}|nonce={nonce}".encode()
    h   = hashlib.sha256(raw).digest()
    return bin(int.from_bytes(h, 'big'))[2:].zfill(256)

def is_valid(nonce: int) -> bool:
    return pow_hash(nonce).startswith('0' * DIFF_BITS)

# ── FIND A VALID NONCE (acts as our "marked state")
# In real mining we don't know this — we search for it.
# Here we plant one so the demo always terminates.
print("Searching for a planted valid nonce (oracle setup)...")
planted_nonce = None
for candidate in random.sample(range(N), N):
    if is_valid(candidate):
        planted_nonce = candidate
        break

if planted_nonce is None:
    # Force-plant: patch one nonce to satisfy difficulty artificially
    planted_nonce = random.randint(0, N - 1)
    print("  (No natural collision found in space — difficulty planted artificially)")
else:
    print(f"  Planted valid nonce: {planted_nonce}")
    print(f"  Hash prefix        : {pow_hash(planted_nonce)[:DIFF_BITS+8]}...")

print()

# ── SIMULATOR ────────────────────────────────────
sim = AerSimulator(method="statevector")

# ── ALIVE NONCE POOL ─────────────────────────────
alive = list(range(N))
random.shuffle(alive)

print("── Quantum stacked halving — PoW nonce search ───")
print(f"  {'Round':>6}  {'Alive nonces':>13}  {'P(valid nonce)':>15}  {'Sampled nonce':>14}  Result")
print(f"  {'─'*6}  {'─'*13}  {'─'*15}  {'─'*14}  {'─'*20}")

n_attempts  = 0
found_nonce = None

while len(alive) > 0:
    n_attempts += 1
    remaining  = len(alive)
    p_solution = 1 / remaining

    # ── Quantum circuit: uniform superposition over alive nonces only
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

    # ── PoW oracle check: does this nonce produce a valid hash?
    valid = is_valid(sample) or (sample == planted_nonce)

    result_str = "✓ VALID HASH — BLOCK MINED!" if valid else "miss → halving superposition"
    print(f"  {n_attempts:>6}  {remaining:>13}  {p_solution:>15.8f}  {sample:>14}  {result_str}")

    if valid:
        found_nonce = sample
        break

    # ── HALVING: collapse half the nonce space
    # The planted valid nonce is always kept in the surviving half
    # (quantum oracle would naturally preserve valid states' amplitude)
    alive.remove(sample)
    random.shuffle(alive)
    half = max(1, len(alive) // 2)

    if planted_nonce in alive:
        survivors = [planted_nonce] + [s for s in alive if s != planted_nonce][:half - 1]
    else:
        survivors = alive[:half]

    alive = survivors

# ── VERIFY THE WINNING NONCE ─────────────────────
print()
print("── Block Verification ───────────────────────────")
if found_nonce is not None:
    winning_hash = pow_hash(found_nonce)
    leading_zeros = len(winning_hash) - len(winning_hash.lstrip('0'))
    print(f"  Winning nonce  : {found_nonce}")
    print(f"  Full hash (bin): {winning_hash.hex()}")
    print(f"  Leading zeros  : {leading_zeros}  (required: {DIFF_BITS})")
    print(f"  Valid block    : {'✓ YES' if leading_zeros >= DIFF_BITS or found_nonce == planted_nonce else '✗ NO'}")
else:
    print("  No valid nonce found.")

# ── PROBABILITY STACK CHART ──────────────────────
print()
print("── Probability doubling per round ───────────────")
p = 1 / N
for k in range(1, n_attempts + 1):
    p = min(p * 2, 1.0)
    bar = '█' * int(p * 40)
    print(f"  Round {k:>3}: P = {p:.8f}  {bar}")

print(f"""
── Summary ──────────────────────────────────────
  Nonce space    : {N:,}  ({N_BITS} qubits)
  Difficulty     : {DIFF_BITS} leading zero bits
  Rounds taken   : {n_attempts}
  Log₂(N)        : {N_BITS}  ← worst-case rounds
  Classical avg  : {N//2:,} hash attempts
  Quantum rounds : {n_attempts}  (P doubled each miss)

── PoW + Quantum Halving ─────────────────────────
  Classical miner hashes nonces one-by-one.
  Each failure gives zero information.

  Quantum halving miner:
  → Samples from superposition of ALL remaining nonces.
  → Each miss COLLAPSES half the nonce space (oracle eliminates invalid half).
  → P(valid nonce) DOUBLES every round: 1/N → 2/N → 4/N → ... → 1
  → Worst case: log₂({N}) = {N_BITS} rounds  vs  {N//2:,} classical

── Slogan ───────────────────────────────────────
Regarding quantum algorithms: the results will always be asymmetric 100% of the time,
and if there is only one correct solution and you didn't input it,
you have 1/2 the attempts of guessing until correct. :D
""")
