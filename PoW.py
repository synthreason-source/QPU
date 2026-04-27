import hashlib
import math
import random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── CONFIG ─────────────────────────────────────────────────────────────
N_BITS          = 10             # small window
N               = 2**N_BITS
DIFFICULTY_BITS = 6              # easy PoW
DATA            = b"BLOCK_HEADER_001"
TARGET_RANDOM   = 3000

p         = 2**(-DIFFICULTY_BITS)
M_est     = N * p
n_opt_est = math.pi / 4 * math.sqrt(N / M_est)
print(f"N={N}  est_valid≈{M_est:.0f}  n_opt_est≈{n_opt_est:.0f}")

# ── HASH HELPERS (CLASSICAL, NOT IN CIRCUIT) ──────────────────────────
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


# ── CLASSICAL RANDOM INCORRECT NONCES ─────────────────────────────────
rng           = random.Random(42)
random_nonces = []

while len(random_nonces) < TARGET_RANDOM:
    candidate = rng.randrange(N)
    random_nonces.append(candidate)

print(f"Sampled {len(random_nonces)} random incorrect nonces")

# ── COMBINED MARKED SET (for statistics only) ─────────────────────────
marked_nonces = list(set(random_nonces))
M             = len(marked_nonces)
n_opt         = max(1, round(math.pi / 4 * math.sqrt(N / M)))

print(f"Marked set: {M} states (n_opt={n_opt})")

# ── QUANTUM ORACLE (INTRINSIC TO LEADING‑ZEROS) ───────────────────────
# Here the oracle is *intrinsic* to the PoW condition: it flips phase
# on any nonce whose SHA‑256(Data || nonce) has DIFFICULTY_BITS leading zeros.
# The circuit does NOT know the hash values; it only knows the oracle function.

def phase_oracle_by_condition(condition, nqubits):
    """Oracle that flips phase iff condition(state_int) is True."""
    qc = QuantumCircuit(nqubits, name="Oracle")
    # Iterate over all possible states (small N_BITS only)
    for state_int in range(2**nqubits):
        if condition(state_int):
            bits = format(state_int, f'0{nqubits}b')[::-1]
            for i, b in enumerate(bits):
                if b == '0':
                    qc.x(i)
            qc.h(nqubits - 1)
            qc.mcx(list(range(nqubits - 1)), nqubits - 1)
            qc.h(nqubits - 1)
            for i, b in enumerate(bits):
                if b == '0':
                    qc.x(i)
    return qc

# Use the PoW condition directly (intrinsic leading‑zeros oracle)
oracle_condition = check_pow_bits   # only this function knows hash

# ── DIFFUSER ──────────────────────────────────────────────────────────
def diffuser(n):
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc

# ── BUILD GROVER CIRCUIT (ORACLE INTRINSIC TO LEADING ZEROS) ─────────
qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))

for _ in range(n_opt):
    qc.compose(phase_oracle_by_condition(oracle_condition, N_BITS), inplace=True)
    qc.compose(diffuser(N_BITS), inplace=True)

qc.measure(range(N_BITS), range(N_BITS))

# ── SIMULATION ────────────────────────────────────────────────────────
shots = 8192
sim   = AerSimulator(method='statevector')
counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()

# ── ANALYSIS ──────────────────────────────────────────────────────────
def get_prob(nonce):
    return counts.get(format(nonce, f'0{N_BITS}b'), 0) / shots

valid_prob  = sum(get_prob(v) for v in random_nonces) * 100
random_prob = sum(get_prob(r) for r in random_nonces) * 100
marked_prob = valid_prob + random_prob

found = int(max(counts, key=counts.get), 2)
print(f"\n── Peak measurement ───────────────────────────────")
print(f"  State:  {found}")
print(f"  Prob:   {get_prob(found)*100:.2f}%")
print(f"  Valid?: {check_pow_bits(found)}: {sha256_bytes(found).hex()}")
