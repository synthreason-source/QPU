import hashlib, random, numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── CONFIG ──────────────────────────────────────────────────────────────────
N_BITS          = 17
DATA            = b"BLOCK_HEADER_001"
DIFFICULTY_BITS = 17
N               = 2**N_BITS
SEARCH_RANGE = 1000
def check_pow_bits(nonce: int) -> bool:
    h = hashlib.sha256(DATA + nonce.to_bytes(4, 'big')).digest()
    num_full_bytes = DIFFICULTY_BITS // 8
    if any(b != 0 for b in h[:num_full_bytes]): return False
    remaining = DIFFICULTY_BITS % 8
    if remaining > 0 and (h[num_full_bytes] >> (8 - remaining)) != 0: return False
    return True

for M in range(N,N+SEARCH_RANGE):
    print(f"Search space: {N} nonces, M={M}")

    # ── Fix: Use a safer `max` approach ───────────────────────────────────────────
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

    n_opt = max(1, round(np.pi / 4 * np.sqrt(N / M)))
    qc = QuantumCircuit(N_BITS, N_BITS)
    qc.h(range(N_BITS))
    for _ in range(n_opt):
        qc.compose(phase_oracle(valid_nonces, N_BITS), inplace=True)
        qc.compose(diffuser(N_BITS),                   inplace=True)
    qc.measure(range(N_BITS), range(N_BITS))

    sim    = AerSimulator(method='statevector')
    counts = sim.run(transpile(qc, sim), shots=2048).result().get_counts()
    # Safely get key with max value
    found = int(max(counts, key=counts.get), 2)
    if check_pow_bits(found):
        final_hash = hashlib.sha256(DATA + found.to_bytes(4, 'big')).hexdigest()
        print(f"Grover found nonce  : {found}")
        print(f"SHA-256(DATA+nonce) : {final_hash}")
        print(f"Verified PoW        : {'✓ VALID' if check_pow_bits(found) else '✗ WRONG'}")