import hashlib
import math
import random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_BITS          = 19
DATA            = b"BLOCK_HEADER_001"
DIFFICULTY_BITS = 11
N               = 2**N_BITS
TARGET_RANDOM   = 37000          # Random incorrect nonces to add to marked set # check out the ratios of true/random the grover algorithm succeeds

p         = 2**(-DIFFICULTY_BITS)
M_est     = N * p
n_opt_est = math.pi / 4 * math.sqrt(N / M_est)
print(f"N={N:,}  est_valid≈{M_est:.0f}  n_opt_est≈{n_opt_est:.0f}")

# ── HASH HELPERS ──────────────────────────────────────────────────────────────
def sha256_bytes(nonce: int) -> bytes:
    return hashlib.sha256(DATA + nonce.to_bytes(4, 'big')).digest()

def check_pow_bits(nonce: int) -> bool:
    h = sha256_bytes(nonce)
    full_bytes = DIFFICULTY_BITS // 8
    if any(b != 0 for b in h[:full_bytes]):
        return False
    rem = DIFFICULTY_BITS % 8
    if rem and (h[full_bytes] >> (8 - rem)) != 0:
        return False
    return True

# ── VALID NONCE ───────────────────────────────────────────────────────────────
print("\nUsing known valid nonce...")
valid_nonces = [n for n in range(N) if check_pow_bits(n)]

print(f"True valid nonces (M={len(valid_nonces)}): {valid_nonces}")

# ── RANDOM INCORRECT NONCES ───────────────────────────────────────────────────
# Plain random nonces that fail the PoW check — no hash structure requirement.
# They go into the same oracle marked set as valid nonces, receiving identical
# phase flips and Grover amplification.
print(f"\nSampling {TARGET_RANDOM} random incorrect nonces...")
valid_set     = set(valid_nonces)
rng           = random.Random(42)
random_nonces = []

while len(random_nonces) < TARGET_RANDOM:
    candidate = rng.randrange(N)
    if candidate not in valid_set and candidate not in set(random_nonces):
        if not check_pow_bits(candidate):
            random_nonces.append(candidate)

print(f"Sampled {len(random_nonces)} random incorrect nonces")

# ── COMBINED MARKED SET ───────────────────────────────────────────────────────
marked_nonces = list(set(valid_nonces + random_nonces))
M             = len(marked_nonces)
n_opt         = max(1, round(math.pi / 4 * math.sqrt(N / M)))

print(f"\nMarked set: {M} total  ({len(valid_nonces)} valid + {len(random_nonces)} random incorrect)")
print(f"n_opt={n_opt}  (adjusted for full marked set size)")

# ── GROVER CIRCUIT ────────────────────────────────────────────────────────────
def phase_oracle(marked: list, n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="Oracle")
    for state in marked:
        bits = format(state, f'0{n}b')[::-1]
        for i, b in enumerate(bits):
            if b == '0':
                qc.x(i)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for i, b in enumerate(bits):
            if b == '0':
                qc.x(i)
    return qc

def diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc

print(f"\nBuilding Grover circuit: {N_BITS} qubits, {n_opt} iterations, {M} marked states...")
qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))

for _ in range(n_opt):
    qc.compose(phase_oracle(marked_nonces, N_BITS), inplace=True)
    qc.compose(diffuser(N_BITS), inplace=True)

qc.measure(range(N_BITS), range(N_BITS))

shots = 65536
sim   = AerSimulator(method='automatic')
print("Running simulation...")
counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()

# ── RESULTS ───────────────────────────────────────────────────────────────────
def get_prob(nonce: int) -> float:
    return counts.get(format(nonce, f'0{N_BITS}b'), 0) / shots

def classify(nonce: int) -> str:
    if check_pow_bits(nonce):           return "✓ VALID"
    if nonce in set(random_nonces):     return "✗ RANDOM (marked)"
    return "✗ unmarked"

print("\n── Top 10 measured states ───────────────────────────────────────────")
top10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
for state_str, cnt in top10:
    state_int = int(state_str, 2)
    print(f"  {state_int:>12,}  {cnt:>6,} shots  {cnt/shots*100:>5.2f}%  {classify(state_int)}")

valid_prob  = sum(get_prob(v) for v in valid_nonces)  * 100
random_prob = sum(get_prob(r) for r in random_nonces) * 100
marked_prob = valid_prob + random_prob

print(f"\n── Probability mass ─────────────────────────────────────────────────")
print(f"  All marked states:      {marked_prob:6.2f}%")
print(f"    True valid:           {valid_prob:6.2f}%  ({len(valid_nonces)} states)")
print(f"    Random incorrect:     {random_prob:6.2f}%  ({len(random_nonces)} states)")
print(f"  Per-state avg (marked): {marked_prob / M:6.3f}%")

print(f"\n── Per-state probabilities ──────────────────────────────────────────")
for v in valid_nonces[:10]:
    print(f"  {v:>12,}  {get_prob(v)*100:>6.3f}%  VALID")
for r in random_nonces[:10]:
    print(f"  {r:>12,}  {get_prob(r)*100:>6.3f}%  RANDOM")
if len(random_nonces) > 10:
    print(f"  ... ({len(random_nonces) - 10} more random nonces not shown)")

found = int(max(counts, key=counts.get), 2)
print(f"\n── Peak measurement ─────────────────────────────────────────────────")
print(f"  State:  {found:,}")
print(f"  Prob:   {get_prob(found)*100:.2f}%")
print(f"  Class:  {classify(found)}")
print(f"  Hash:   {sha256_bytes(found).hex()}")

if check_pow_bits(found):
    print(f"\n  ✓ Accepted by PoW verifier")
else:
    print(f"\n  ✗ Rejected by PoW verifier (random decoy was amplified equally)")
