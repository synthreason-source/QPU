#!/usr/bin/env python3
"""
TOY HASH PHASE-INTERFERENCE / GROVER SIMULATION

This is a numerical model, not optical hardware and not SHA-256.

An N x N plane represents M=N^2 candidates, with an exactly uniform
initial state psi[i,j] = 1/N. The oracle is a lossless phase flip and the
diffuser is inversion about the global mean.

Important design choice:
- The selectable target is an N-bit *display digest*.
- The oracle marks the candidate(s) whose N-bit digest matches that target.
- Collisions are permitted and explicitly reported: Grover then amplifies all
  marked preimages, not necessarily the chosen `secret`.
- `--require-unique` fails early if the target has more than one preimage.

The toy mixing function is deterministic and width-scalable. It is NOT
cryptographically secure, not SHA-256 compatible, and not intended for
preimage resistance claims.
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
# TOY HASH
# ----------------------------------------------------------------------

def toy_sha_variable(x: int, bits: int) -> int:
    """Return an N-bit SHA-inspired non-cryptographic digest.

    It intentionally accepts a finite-width integer and returns exactly
    `bits` output bits. It is deterministic, but collisions are expected:
    any N-bit hash has at most 2^N possible outputs.
    """
    mask = mask_for_bits(bits)
    x &= mask

    # Domain-separate small widths before width reduction.
    a = (0x6A09E667F3BCC908 ^ x) & mask
    b = (0xBB67AE8584CAA73B ^ rotl(x, 1, bits)) & mask
    c = (0x3C6EF372FE94F82B ^ rotr(x, 3, bits)) & mask
    d = (0xA54FF53A5F1D36F1 ^ ((x + 0x5A) & mask)) & mask

    w = [
        x,
        rotl(x ^ 0x36, 1, bits),
        rotr(x ^ 0xA5, 3, bits),
        (x + 0x5A) & mask,
    ]
    rounds = max(12, bits // 2)

    for r in range(rounds):
        k = (0x9E3779B97F4A7C15 * (r + 1) + 0xD1B54A32D192ED03) & mask
        wi = w[r & 3]
        if r & 1:
            f = ((a & b) ^ (a & c) ^ (b & c)) & mask
            sigma = rotr(c, 7, bits) ^ rotr(c, 11, bits) ^ rotl(c, 3, bits)
        else:
            f = ((b & c) ^ ((~b) & d)) & mask
            sigma = rotr(b, 5, bits) ^ rotr(b, 9, bits) ^ rotl(b, 4, bits)

        t1 = (d + sigma + f + wi + k) & mask
        t2 = (rotl(a, 3, bits) ^ rotr(a, 2, bits) ^ rotl(c, 1, bits)) & mask
        a, b, c, d = (t1 + t2) & mask, a, b, c
        w[r & 3] = (wi ^ rotl(w[(r + 1) & 3], 1, bits) ^ k ^ a) & mask

    return (a ^ rotl(b, 1, bits) ^ rotr(c, 2, bits) ^ rotl(d, 3, bits)) & mask


def sha256_bits(x: int, input_bits: int = 8) -> int:
    """Reference real SHA-256 digest; unused by the oracle."""
    byte_count = max(1, (input_bits + 7) // 8)
    return int.from_bytes(hashlib.sha256(x.to_bytes(byte_count, "big")).digest(), "big")


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
    return np.stack([integer_to_bits(toy_sha_variable(x, N), N) for x in range(N * N)])


def build_target_comparison(N: int, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = N * N
    target &= mask_for_bits(N)
    hashes = np.fromiter((toy_sha_variable(x, N) for x in range(candidates)), dtype=object, count=candidates)
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
    plot_matrix(axes[1, 1], hash_bits, f"TOY HASH BITS ({candidates} x {N})", "Greys")
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
        "", "This is a non-cryptographic toy", "hash; collisions are expected.",
    ]), va="top", family="monospace", fontsize=11)
    axes[2, 2].text(0.02, 0.95, "\n".join([
        "ORACLE", "", f"Target: 0x{target:X}", f"Preimages: {marked}", "", "Non-match: +1 phase", "Match: -1 phase", "", "Diffuser: 2*mean(state)-state",
    ]), va="top", family="monospace", fontsize=11)
    axes[2, 3].text(0.02, 0.95, "\n".join([
        "MEASUREMENT", "", f"Best candidate: {best}", f"Position: ({best // N}, {best % N})",
        f"Hash(best): 0x{int(result['hashes'][best]):X}", f"P(best): {probability.reshape(-1)[best]:.8f}",
        f"P(marked set): {probability.reshape(-1)[matches].sum():.8f}",
    ]), va="top", family="monospace", fontsize=11)
    fig.suptitle(f"UNIFORM {N}x{N} PLANE -> {N}-BIT TOY HASH -> PHASE ORACLE -> GROVER", fontsize=16)
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
    target = toy_sha_variable(secret, N) if target is None else target & mask_for_bits(N)

    field = make_equal_amplitude_field(N)
    binary_valves = build_binary_filter(N, N, seed=101)
    binary_encoded = encode_same_plane(np.ones((N, N)), binary_valves)
    joint_power, balanced_power, cross_correlation = balanced_interference(field, np.roll(field, 1, axis=1))
    result = optical_grover(N, target, iterations=iterations, require_unique=require_unique)

    marked_preimages = np.flatnonzero(result["matches"]).tolist()
    secret_marked = bool(result["matches"][secret])
    best = result["best"]
    print("=" * 76)
    print("UNIFORM-AMPLITUDE TOY-HASH PHASE-INTERFERENCE / GROVER MODEL")
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
        out_path = f"toy_sha_interference_{N}x{N}.png"
    make_visualization(N, target, result, out_path)
    print(f"\nSaved visualization: {out_path}")
    return {"field": field, "binary_valves": binary_valves, "binary_encoded": binary_encoded,
            "joint_power": joint_power, "balanced_power": balanced_power,
            "cross_correlation": cross_correlation, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform-amplitude toy-hash phase-interference/Grover simulation.")
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
