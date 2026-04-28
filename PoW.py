import random
import math
import hashlib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── CONFIG ───────────────────────────────────────
N_BITS    = 24
N         = 2**N_BITS
DIFF_BITS = 22

BLOCK_HEADER = "First quantum sha256 by George W 28-4-2026"

print(f"N={N}  ({N_BITS}-qubit nonce space)")
print(f"Block header : {BLOCK_HEADER}")
print(f"Difficulty   : {DIFF_BITS} leading zero bits  (1-in-{2**DIFF_BITS:,})")
print(f"Valid nonces : unknown — search is blind")
print()

# ── PoW HASH FUNCTIONS ────────────────────────────
def pow_hash_hex(nonce: int) -> str:
    raw = f"{BLOCK_HEADER}|nonce={nonce}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def leading_zero_bits(hex_hash: str) -> int:
    val = int(hex_hash, 16)
    if val == 0:
        return 256
    return 255 - val.bit_length() + 1

def is_valid(nonce: int) -> bool:
    return leading_zero_bits(pow_hash_hex(nonce)) >= DIFF_BITS

def half_contains_valid(candidates: list) -> bool:
    """Oracle: check if ANY nonce in this half is valid. O(n) but honest."""
    return any(is_valid(n) for n in candidates)

# ── SIMULATOR ────────────────────────────────────
sim = AerSimulator(method="statevector")

# ── ALIVE NONCE POOL ─────────────────────────────
alive = list(range(N))
random.shuffle(alive)

print("── Quantum stacked halving — blind PoW search ───")
print(f"  {'Round':>6}  {'Alive':>10}  {'P(hit)':>14}  {'Sampled':>10}  {'Hash (hex)':>64}  {'Zeros':>5}  Result")
print(f"  {'─'*6}  {'─'*10}  {'─'*14}  {'─'*10}  {'─'*64}  {'─'*5}  {'─'*22}")

n_attempts  = 0
found_nonce = None

while len(alive) > 0:
    n_attempts += 1
    remaining  = len(alive)
    p_solution = 1 / remaining

    # ── Quantum circuit: uniform superposition over alive nonces
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

    sample_hex = pow_hash_hex(sample)
    lz         = leading_zero_bits(sample_hex)
    valid      = lz >= DIFF_BITS

    result_str = "✓ VALID — BLOCK MINED!" if valid else "miss → halving"
    print(f"  {n_attempts:>6}  {remaining:>10}  {p_solution:>14.10f}  {sample:>10}  {sample_hex}  {lz:>5}  {result_str}")

    if valid:
        found_nonce = sample
        break

    # ── HALVING with oracle verification on the cut
    # Remove the sampled miss, shuffle, split into two halves
    alive.remove(sample)
    random.shuffle(alive)
    half = max(1, len(alive) // 2)

    keep_half  = alive[:half]
    drop_half  = alive[half:]

    # Oracle checks the DROP half — if valid nonce is there, swap
    # This is the key: we verify AFTER the cut, not before
    if drop_half and not half_contains_valid(drop_half):
        alive = drop_half   # valid nonce is in the half we were about to discard
    else:
        alive = drop_half   # valid nonce is in the keep half (or doesn't exist)

# ── FULL BLOCK VERIFICATION ───────────────────────
print()
print("── Block Verification ───────────────────────────────────────────────────────────────")
if found_nonce is not None:
    raw_in   = f"{BLOCK_HEADER}|nonce={found_nonce}"
    hex_hash = pow_hash_hex(found_nonce)
    bin_hash = bin(int(hex_hash, 16))[2:].zfill(256)
    lz       = leading_zero_bits(hex_hash)
    valid    = lz >= DIFF_BITS

    print(f"  Input         : {raw_in}")
    print(f"  SHA-256 (hex) : {hex_hash}")
    print(f"  SHA-256 (bin) : {bin_hash[:64]}")
    print(f"                  {bin_hash[64:128]}")
    print(f"                  {bin_hash[128:192]}")
    print(f"                  {bin_hash[192:256]}")
    print(f"  Leading zeros : {lz} bits")
    print(f"  Difficulty    : {DIFF_BITS} bits required")
    print(f"  Verified      : {'✓ VALID BLOCK' if valid else '✗ INVALID'}")
else:
    print("  No valid nonce in search space.")

# ── PROBABILITY STACK CHART ──────────────────────
print()
print("── Probability doubling per round ───────────────")
p = 1 / N
for k in range(1, n_attempts + 1):
    p = min(p * 2, 1.0)
    bar = '█' * int(p * 50)
    print(f"  Round {k:>3}: P = {p:.10f}  {bar}")

print(f"""
── Summary ──────────────────────────────────────
  Block header   : {BLOCK_HEADER}
  Nonce space    : {N:,}  ({N_BITS} qubits)
  Difficulty     : {DIFF_BITS} leading zero bits
  Rounds taken   : {n_attempts}
  Result         : {'MINED nonce=' + str(found_nonce) if found_nonce else 'NOT FOUND'}
  Log₂(N)        : {N_BITS}  ← worst-case rounds

── Slogan ───────────────────────────────────────
Regarding quantum algorithms: the results will always be asymmetric 100% of the time,
and if there is only one correct solution and you didn't input it,
you have 1/2 the attempts of guessing until correct. :D
""")
