#!/usr/bin/env python3
"""
REAL SHA-256 PHASE-INTERFERENCE / GROVER SIMULATION

This is a numerical model, not optical hardware.

An N x N plane represents M=N^2 candidates, with an exactly uniform
initial state psi[i,j] = 1/N. The oracle is a lossless phase flip and the
diffuser is inversion about the global mean.

The hash function is real SHA-256 (truncated to N bits for the oracle).
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def mask_for_bits(bits: int) -> int:
    if bits < 1:
        raise ValueError("bits must be >= 1")
    return (1 << bits) - 1


def rotl(x: int, n: int, bits: int) -> int:
    mask = mask_for_bits(bits)
    n %= bits
    x &= mask
    return x if n == 0 else ((x << n) | (x >> (bits - n))) & mask


def rotr(x: int, n: int, bits: int) -> int:
    mask = mask_for_bits(bits)
    n %= bits
    x &= mask
    return x if n == 0 else ((x >> n) | (x << (bits - n))) & mask


# ----------------------------------------------------------------------
# REAL SHA-256 IMPLEMENTATION
# ----------------------------------------------------------------------

def sha256_compress(state: list[int], chunk: bytes) -> list[int]:
    """Pure-Python SHA-256 compression function for a single 64-byte block."""
    K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ]

    def rotr32(x, n):
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    def ch(x, y, z):
        return (x & y) ^ ((~x) & z)

    def maj(x, y, z):
        return (x & y) ^ (x & z) ^ (y & z)

    def sigma0(x):
        return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)

    def sigma1(x):
        return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)

    def gamma0(x):
        return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3)

    def gamma1(x):
        return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10)

    W = list(int.from_bytes(chunk[i*4:(i+1)*4], 'big') for i in range(16))
    for i in range(16, 64):
        W.append((gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16]) & 0xFFFFFFFF)

    a, b, c, d, e, f, g, h = state

    for i in range(64):
        T1 = (h + sigma1(e) + ch(e, f, g) + K[i] + W[i]) & 0xFFFFFFFF
        T2 = (sigma0(a) + maj(a, b, c)) & 0xFFFFFFFF
        h = g
        g = f
        f = e
        e = (d + T1) & 0xFFFFFFFF
        d = c
        c = b
        b = a
        a = (T1 + T2) & 0xFFFFFFFF

    return [
        (state[0] + a) & 0xFFFFFFFF,
        (state[1] + b) & 0xFFFFFFFF,
        (state[2] + c) & 0xFFFFFFFF,
        (state[3] + d) & 0xFFFFFFFF,
        (state[4] + e) & 0xFFFFFFFF,
        (state[5] + f) & 0xFFFFFFFF,
        (state[6] + g) & 0xFFFFFFFF,
        (state[7] + h) & 0xFFFFFFFF,
    ]


def sha256_from_int(x: int, input_bits: int) -> int:
    """Compute real SHA-256 of an integer, returning full 256-bit digest."""
    byte_len = max(1, (input_bits + 7) // 8)
    data = x.to_bytes(byte_len, 'big')
    msg_len_bits = len(data) * 8
    padded = bytearray(data)
    padded.append(0x80)
    while (len(padded) % 64) != 56:
        padded.append(0x00)
    padded.extend(msg_len_bits.to_bytes(8, 'big'))
    H = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]
    H = sha256_compress(H, bytes(padded))
    digest = 0
    for h in H:
        digest = (digest << 32) | h
    return digest


def sha256_bits(x: int, input_bits: int, hash_bits: int) -> int:
    """Return SHA-256 digest truncated to `hash_bits`."""
    full = sha256_from_int(x, input_bits)
    if hash_bits < 256:
        full >>= (256 - hash_bits)
    return full & mask_for_bits(hash_bits)


# ----------------------------------------------------------------------
# FIELD / HARDWARE DIAGNOSTICS
# ----------------------------------------------------------------------

def make_equal_amplitude_field(N: int) -> np.ndarray:
    if N < 1:
        raise ValueError("N must be >= 1")
    return np.full((N, N), 1.0 / N, dtype=np.complex128)


def field_norm(field: np.ndarray) -> float:
    return float(np.vdot(field, field).real)


def build_binary_filter(rows: int, cols: int, seed: int, p_open: float = 0.5) -> np.ndarray:
    if not 0.0 <= p_open <= 1.0:
        raise ValueError("p_open must lie in [0, 1]")
    return (np.random.default_rng(seed).random((rows, cols)) < p_open).astype(float)


def encode_same_plane(values: np.ndarray, valves: np.ndarray) -> np.ndarray:
    if values.shape != valves.shape:
        raise ValueError("values and valves must have identical shapes")
    if not np.all((valves == 0.0) | (valves == 1.0)):
        raise ValueError("valves must be binary")
    return values * valves


def apply_phase_plane(field: np.ndarray, phase: np.ndarray) -> np.ndarray:
    if field.shape != phase.shape:
        raise ValueError("field and phase must have identical shapes")
    if not np.allclose(np.abs(phase), 1.0):
        raise ValueError("phase plane must be unit modulus")
    return field * phase


def fft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def ifft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))


def balanced_interference(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    F_A, F_B = fft2c(A), fft2c(B)
    joint_power = np.abs(F_A + F_B) ** 2
    balanced_power = joint_power - np.abs(F_A) ** 2 - np.abs(F_B) ** 2
    cross_spectrum = F_A * np.conj(F_B) + np.conj(F_A) * F_B
    return joint_power, balanced_power, np.real(ifft2c(cross_spectrum))


# ----------------------------------------------------------------------
# HASH TABLE / ORACLE
# ----------------------------------------------------------------------

def integer_to_bits(value: int, bits: int) -> np.ndarray:
    return np.fromiter(((value >> b) & 1 for b in range(bits)), dtype=np.uint8, count=bits)


def build_candidate_bit_planes(N: int) -> np.ndarray:
    return np.stack([integer_to_bits(x, N) for x in range(N * N)])


def build_hash_bit_planes(N: int) -> np.ndarray:
    return np.stack([integer_to_bits(sha256_bits(x, N, N), N) for x in range(N * N)])


def build_target_comparison(N: int, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = N * N
    target &= mask_for_bits(N)
    hashes = np.fromiter((sha256_bits(x, N, N) for x in range(candidates)), dtype=object, count=candidates)
    xor_values = np.fromiter((int(h) ^ target for h in hashes), dtype=object, count=candidates)
    matches = np.equal(xor_values, 0)
    phase = np.where(matches, -1.0 + 0.0j, 1.0 + 0.0j)
    return hashes, xor_values, matches, phase


def hash_statistics(hashes: np.ndarray) -> dict[str, int]:
    counts = Counter(map(int, hashes))
    multiplicities = list(counts.values())
    return {
        "distinct_digests": len(counts),
        "colliding_digests": sum(n > 1 for n in multiplicities),
        "max_preimages": max(multiplicities),
    }


# ----------------------------------------------------------------------
# GROVER AMPLITUDE AMPLIFICATION
# ----------------------------------------------------------------------

def diffusion_2d(field: np.ndarray) -> np.ndarray:
    return 2.0 * np.mean(field) - field


def recommended_iterations(candidates: int, marked_count: int) -> int:
    theta = math.asin(math.sqrt(marked_count / candidates))
    return max(0, int(round(math.pi / (4.0 * theta) - 0.5)))


def optical_grover(N: int, target: int, iterations: int | None = None, require_unique: bool = False) -> dict:
    candidates = N * N
    hashes, xor_values, matches, phase = build_target_comparison(N, target)
    marked_count = int(matches.sum())
    if marked_count == 0:
        raise RuntimeError("Target has no preimage in the selected N^2 search space.")
    if require_unique and marked_count != 1:
        preimages = np.flatnonzero(matches).tolist()
        raise RuntimeError(f"Target has {marked_count} preimages in the search space: {preimages}")

    if iterations is None:
        iterations = recommended_iterations(candidates, marked_count)

    state = make_equal_amplitude_field(N)
    initial_norm = field_norm(state)
    phase_2d = phase.reshape(N, N)
    history = []
    for _ in range(iterations):
        state = diffusion_2d(apply_phase_plane(state, phase_2d))
        history.append(float(np.abs(state.reshape(-1)[matches]) @ np.abs(state.reshape(-1)[matches])))

    probability = np.abs(state) ** 2
    best = int(np.argmax(probability.reshape(-1)))
    return {
        "state": state, "probability": probability, "hashes": hashes,
        "xor_values": xor_values, "matches": matches, "phase": phase,
        "history": history, "iterations": iterations, "marked_count": marked_count,
        "best": best, "initial_norm": initial_norm, "final_norm": field_norm(state),
        "hash_stats": hash_statistics(hashes),
    }


# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------

def plot_matrix(ax, data: np.ndarray, title: str, cmap: str = "viridis"):
    image = ax.imshow(data, origin="lower", interpolation="nearest", aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    return image


def make_visualization(N: int, target: int, result: dict, out_path: str) -> None:
    candidates = N * N
    candidate_bits = build_candidate_bit_planes(N).astype(float)
    hash_bits = build_hash_bit_planes(N).astype(float)
    matches = result["matches"]
    probability = result["probability"]
    xor_log = np.array([math.log2(int(x) + 1) for x in result["xor_values"]], dtype=float)

    fig, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    plot_matrix(axes[0, 0], np.ones((N, N)), f"{N}x{N} UNIFORM AMPLITUDE", "Greys")
    plot_matrix(axes[0, 1], matches.reshape(N, N).astype(float), "TARGET PREIMAGE PLANE", "Greys")
    plot_matrix(axes[0, 2], result["phase"].reshape(N, N).real, "PHASE ORACLE (+1 / -1)", "coolwarm")
    plot_matrix(axes[0, 3], probability, "FINAL PROBABILITY", "inferno")
    plot_matrix(axes[1, 0], candidate_bits, f"CANDIDATE BITS ({candidates} x {N})", "Greys")
    plot_matrix(axes[1, 1], hash_bits, f"SHA-256 BITS ({candidates} x {N})", "Greys")
    plot_matrix(axes[1, 2], xor_log.reshape(N, N), "HASH XOR TARGET: log2(x+1)", "inferno")

    ax = axes[1, 3]
    ax.plot(range(1, len(result["history"]) + 1), result["history"], marker="o")
    ax.set_title("MARKED-SET PROBABILITY")
    ax.set_xlabel("Grover iteration")
    ax.set_ylabel("P(all marked preimages)")
    ax.grid(True, alpha=0.25)

    for ax in axes[2]:
        ax.axis("off")
    stats = result["hash_stats"]
    best = result["best"]
    marked = np.flatnonzero(matches).tolist()
    axes[2, 0].text(0.02, 0.95, "\n".join([
        "ARCHITECTURE", "", f"Plane: {N} x {N}", f"Candidates: {candidates:,}",
        f"Digest width: {N} bits", f"Initial amplitude: 1/{N}",
        f"Initial norm: {result['initial_norm']:.12f}", f"Final norm: {result['final_norm']:.12f}",
        f"Marked preimages: {result['marked_count']}", f"Grover rounds: {result['iterations']}",
    ]), va="top", family="monospace", fontsize=11)
    axes[2, 1].text(0.02, 0.95, "\n".join([
        "HASH TABLE DIAGNOSTICS", "", f"Distinct digests: {stats['distinct_digests']}",
        f"Colliding digests: {stats['colliding_digests']}", f"Maximum preimages: {stats['max_preimages']}",
        "", "This uses real SHA-256", "(truncated to N bits).",
    ]), va="top", family="monospace", fontsize=11)
    axes[2, 2].text(0.02, 0.95, "\n".join([
        "ORACLE", "", f"Target: 0x{target:X}", f"Preimages: {marked}", "", "Non-match: +1 phase", "Match: -1 phase", "", "Diffuser: 2*mean(state)-state",
    ]), va="top", family="monospace", fontsize=11)
    axes[2, 3].text(0.02, 0.95, "\n".join([
        "MEASUREMENT", "", f"Best candidate: {best}", f"Position: ({best // N}, {best % N})",
        f"Hash(best): 0x{int(result['hashes'][best]):X}", f"P(best): {probability.reshape(-1)[best]:.8f}",
        f"P(marked set): {probability.reshape(-1)[matches].sum():.8f}",
    ]), va="top", family="monospace", fontsize=11)
    fig.suptitle(f"UNIFORM {N}x{N} PLANE -> {N}-BIT SHA-256 -> PHASE ORACLE -> GROVER", fontsize=16)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# RUNNER / CLI
# ----------------------------------------------------------------------

def run(N: int = 8, secret: int = 37, target: int | None = None, out_path: str | None = None,
        iterations: int | None = None, require_unique: bool = False) -> dict:
    if N < 2:
        raise ValueError("N must be >= 2")
    candidates = N * N
    if not 0 <= secret < candidates:
        raise ValueError(f"secret must be in [0, {candidates - 1}]")
    target = sha256_bits(secret, N, N) if target is None else target & mask_for_bits(N)

    field = make_equal_amplitude_field(N)
    binary_valves = build_binary_filter(N, N, seed=101)
    binary_encoded = encode_same_plane(np.ones((N, N)), binary_valves)
    joint_power, balanced_power, cross_correlation = balanced_interference(field, np.roll(field, 1, axis=1))
    result = optical_grover(N, target, iterations=iterations, require_unique=require_unique)

    marked_preimages = np.flatnonzero(result["matches"]).tolist()
    secret_marked = bool(result["matches"][secret])
    best = result["best"]
    print("=" * 76)
    print("UNIFORM-AMPLITUDE SHA-256 PHASE-INTERFERENCE / GROVER MODEL")
    print("=" * 76)
    print(f"Optical plane:          {N} x {N}")
    print(f"Spatial candidates:     {candidates:,}")
    print(f"Digest width:           {N} bits")
    print(f"Secret candidate:       {secret}")
    print(f"Target digest:          0x{target:X}")
    print(f"Initial amplitude:      {1.0/N:.12f}")
    print(f"Initial norm:           {field_norm(field):.12f}")
    print(f"Final norm:             {result['final_norm']:.12f}")
    print(f"Valve open fraction:    {binary_valves.mean():.4f} (diagnostic only)")
    print(f"Balanced power min/max: {balanced_power.min():.6f} / {balanced_power.max():.6f}")
    print(f"Cross-correlation max:  {np.max(np.abs(cross_correlation)):.6f}")
    print(f"Marked preimages:       {result['marked_count']} -> {marked_preimages}")
    print(f"Secret is marked:       {secret_marked}")
    print(f"Grover iterations:      {result['iterations']}")
    print(f"Best candidate:         {best}")
    print(f"Hash(best):             0x{int(result['hashes'][best]):X}")
    print(f"P(best):                {result['probability'].reshape(-1)[best]:.8f}")
    print(f"P(all marked):          {result['probability'].reshape(-1)[result['matches']].sum():.8f}")
    print(f"Distinct digests:       {result['hash_stats']['distinct_digests']}")
    print(f"Max digest preimages:   {result['hash_stats']['max_preimages']}")
    print("\nINTERFERENCE TRAJECTORY")
    for i, p in enumerate(result["history"], 1):
        print(f"  round {i:3d}: P(marked set) = {p:.8f}")

    if out_path is None:
        out_path = f"toy_sha256_interference_{N}x{N}.png"
    make_visualization(N, target, result, out_path)
    print(f"\nSaved visualization: {out_path}")
    return {"field": field, "binary_valves": binary_valves, "binary_encoded": binary_encoded,
            "joint_power": joint_power, "balanced_power": balanced_power,
            "cross_correlation": cross_correlation, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform-amplitude SHA-256 phase-interference/Grover simulation.")
    parser.add_argument("--size", type=int, default=8, help="N in the N x N plane and N-bit digest.")
    parser.add_argument("--secret", type=int, default=37, help="Candidate used to derive target when --target is omitted.")
    parser.add_argument("--target", type=lambda s: int(s, 0), default=None, help="Target N-bit digest, e.g. 0x5A.")
    parser.add_argument("--iterations", type=int, default=None, help="Override recommended Grover iteration count.")
    parser.add_argument("--require-unique", action="store_true", help="Fail unless exactly one candidate matches target.")
    parser.add_argument("--output", type=str, default=None, help="PNG output path.")
    args = parser.parse_args()
    run(args.size, args.secret, args.target, args.output, args.iterations, args.require_unique)


if __name__ == "__main__":
    main()
