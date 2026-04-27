import hashlib, numpy as np
import math
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── CONFIG ───────────────────────────────────────────────────────────────────
N_BITS          = 20      # 1,048,576 space (feasible sim)
DATA            = b"BLOCK_HEADER_001"
DIFFICULTY_BITS = 18      # Target ~4 solutions
N               = 2**N_BITS
p               = 2**(-DIFFICULTY_BITS)
M_est           = N * p
n_opt_est       = math.pi/4 * math.sqrt(N/M_est)
print(f"N={N:,}, est M={M_est}, n_opt≈{n_opt_est:.0f}")

def check_pow_bits(nonce):
    """Verify first DIFFICULTY_BITS hash bits == 0."""
    h = hashlib.sha256(DATA + nonce.to_bytes(4, 'big')).digest()
    num_full_bytes = DIFFICULTY_BITS // 8
    if any(b != 0 for b in h[:num_full_bytes]): return False
    remaining = DIFFICULTY_BITS % 8
    if remaining > 0 and (h[num_full_bytes] >> (8 - remaining)) != 0: return False
    return True

print("Finding real valid nonces classically...")
valid_nonces = [n for n in range(N) if check_pow_bits(n)]
print(f"Real M={len(valid_nonces)}: {sorted(valid_nonces)}")

# FIXED EMBED: Pre-allocate pool larger than needed
random.seed(42)  # Reproducible
incorrect_candidates = [n for n in range(N) if n not in set(valid_nonces)]
random.shuffle(incorrect_candidates)
base_incorrect = incorrect_candidates[:70]  # FIXED: Enough for 60 total (56 incorrect + 4 valid)

# Embed: create 60-item list with 4 valids at random positions
target_len = 60
positions = sorted(random.sample(range(target_len), min(4, len(valid_nonces))))
embedded_incorrect = []
valid_idx = 0
for i in range(target_len):
    if valid_idx < len(positions) and i == positions[valid_idx]:
        embedded_incorrect.append(valid_nonces[valid_idx])
        valid_idx += 1
    else:
        # FIXED: Check base_incorrect not empty
        if base_incorrect:
            embedded_incorrect.append(base_incorrect.pop(0))
        else:
            embedded_incorrect.append(random.choice(incorrect_candidates))  # Fallback

print(f"Embedded incorrect nonces (len={len(embedded_incorrect)}, {sum(check_pow_bits(n) for n in embedded_incorrect)} hidden valids): {embedded_incorrect[:15]}...")

# Worst from embedded (mixes true-worst + hidden valids)
print("Computing hashes for embedded worst...")
hashes = [(nonce, int(hashlib.sha256(DATA + nonce.to_bytes(4, 'big')).hexdigest(), 16)) 
          for nonce in embedded_incorrect]
hashes.sort(key=lambda x: x[1], reverse=True)
worst_nonces = [nonce for nonce, _ in hashes[:10]]
true_worst_count = sum(not check_pow_bits(w) for w in worst_nonces)
print(f"10 embedded worst nonces: {worst_nonces}")
print(f"True incorrect among worst: {true_worst_count}/10")

def phase_oracle(marked, n):
    qc = QuantumCircuit(n, name="SHA_Oracle")
    for state in marked:
        bits = format(state, f'0{n}b')[::-1]
        for i, b in enumerate(bits):
            if b == '0': qc.x(i)
        qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1)
        for i, b in enumerate(bits):
            if b == '0': qc.x(i)
    return qc

def diffuser(n):
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n)); qc.x(range(n))
    qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1)
    qc.x(range(n)); qc.h(range(n))
    return qc

# Grover on true valids
M = len(valid_nonces)
n_opt = max(1, round(math.pi / 4 * np.sqrt(N / M)))
print(f"n_opt={n_opt} for true M={M}")

qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))
for _ in range(n_opt):
    qc.compose(phase_oracle(valid_nonces, N_BITS), inplace=True)
    qc.compose(diffuser(N_BITS), inplace=True)
qc.measure(range(N_BITS), range(N_BITS))

shots = 65536
sim = AerSimulator(method='automatic')
print("Running Grover...")
counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()

def get_prob(state_int, counts, shots, n_bits):
    return counts.get(format(state_int, f'0{n_bits}b'), 0) / shots

found = int(max(counts, key=counts.get), 2)
prob_found = get_prob(found, counts, shots, N_BITS)

top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 states:")
for state_str, cnt in top5:
    state_int = int(state_str, 2)
    prob = cnt / shots * 100
    status = " ✓" if check_pow_bits(state_int) else " ✗"
    print(f"  {state_int:,} ({cnt:,}, {prob:.2f}%){status}")

print(f"\nPeak: {found:,} ({prob_found*100:.2f}%)")

valid_total = sum(get_prob(v, counts, shots, N_BITS) for v in valid_nonces) * 100
embedded_total = sum(get_prob(e, counts, shots, N_BITS) for e in embedded_incorrect) * 100
worst_total = sum(get_prob(w, counts, shots, N_BITS) for w in worst_nonces) * 100
print(f"True valids: {valid_total:.2f}%")
print(f"Embedded incorrect: {embedded_total:.2f}% (hides valids)")
print(f"Worst nonces: {worst_total:.2f}%")

if check_pow_bits(found):
    h = hashlib.sha256(DATA + found.to_bytes(4, 'big')).hexdigest()
    print(f"\n✓ VALID: {found:,} | {h}")
