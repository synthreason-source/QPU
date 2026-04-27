import hashlib, random, numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_BITS, HASH_BITS, N = 12, 16, 128

def sha_hash(x: int) -> int:
    raw = hashlib.sha256(x.to_bytes(4, 'big')).digest()
    return int.from_bytes(raw[:4], 'big') >> (32 - HASH_BITS)

def phase_oracle(marked, n):
    qc = QuantumCircuit(n, name="SHA_Oracle")
    for state in marked:
        bits = format(state, f'0{n}b')[::-1]   # LSB-first for Qiskit
        for i, b in enumerate(bits):
            if b == '0': qc.x(i)
        qc.h(n-1)
        qc.mcx(list(range(n-1)), n-1)   # MCX = phase flip on |111...1⟩
        qc.h(n-1)
        for i, b in enumerate(bits):
            if b == '0': qc.x(i)
    return qc

def diffuser(n):
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n)); qc.x(range(n))
    qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1)
    qc.x(range(n)); qc.h(range(n))
    return qc

# --- Setup ---
secret      = 64
target_hash = sha_hash(secret)
preimages   = [x for x in range(N) if sha_hash(x) == target_hash]
M           = len(preimages)
if M == 0:
    raise ValueError(
        f"No preimage found for secret={secret} (hash={target_hash:#04x}). "
        f"Ensure 0 <= secret < N={N}."
    )
n_opt       = max(1, round(np.pi/4 * np.sqrt(N / M)))  # optimal iterations

# --- Build & Run ---
qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))
for _ in range(n_opt):
    qc.compose(phase_oracle(preimages, N_BITS), inplace=True)
    qc.compose(diffuser(N_BITS),                inplace=True)
qc.measure(range(N_BITS), range(N_BITS))

counts = AerSimulator(method='statevector').run(
    transpile(qc, AerSimulator()), shots=2048).result().get_counts()

found = max({int(k, 2): v for k, v in counts.items()}, key=lambda x: counts.get(format(x, f'0{N_BITS}b'), 0))
print(f"Found: {found}  SHA256={sha_hash(found):#04x}  Match={found in preimages}")
