import hashlib
import math
import random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── CONFIG ─────────────────────────────────────────────────────────────
N_BITS          = 12              # small window
N               = 2**N_BITS
DIFFICULTY_BITS = 6               # easy PoW
DATA            = b"BLOCK_HEADER_001"
TARGET_RANDOM = 60
p         = 2**(-DIFFICULTY_BITS)
M_est     = N * p
n_opt_est = math.pi / 4 * math.sqrt(N / M_est)
print(f"N={N}  est_valid≈{M_est:.0f}  n_opt_est≈{n_opt_est:.0f}")

# ── HASH HELPERS ───────────────────────────────────────────────────────
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

# ── VALID NONCES (small range) ────────────────────────────────────────

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
valid_nonces = list(set(valid_nonces + random_nonces))

print(f"Valid nonces (M={len(valid_nonces)}): {valid_nonces}")

# ── GROVER CIRCUIT (only valid_nonces marked) ─────────────────────────
def phase_oracle(marked, n):
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

M        = len(valid_nonces)
if M == 0:
    print("No valid nonces in this range; increase difficulty or window.")
else:
    n_opt = max(1, round(math.pi / 4 * math.sqrt(N / M)))

    print(f"Marked set: {M} states, n_opt={n_opt}")

    qc = QuantumCircuit(N_BITS, N_BITS)
    qc.h(range(N_BITS))
    for _ in range(n_opt):
        qc.compose(phase_oracle(valid_nonces, N_BITS), inplace=True)
        qc.compose(diffuser(N_BITS), inplace=True)
    qc.measure(range(N_BITS), range(N_BITS))

    # Faster simulation
    shots = 8192
    sim = AerSimulator(method='statevector')  # or 'automatic'
    counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()

    # Analysis
    def get_prob(nonce):
        return counts.get(format(nonce, f'0{N_BITS}b'), 0) / shots

    valid_prob  = sum(get_prob(v) for v in valid_nonces) * 100
    print(f"Valid states total probability: {valid_prob:.2f}%")
    if valid_prob > 0:
        found = int(max(counts, key=counts.get), 2)
        print(f"Peak: {found}, PoW accepted: {check_pow_bits(found)}")
