#!/usr/bin/env python3
"""
REAL SHA-256 ORACLE / GROVER-AMPLIFICATION SIMULATOR

This is a classical numerical simulation of amplitude amplification, not a
quantum or optical implementation. SHA-256 is evaluated classically to build
the oracle table; a real Grover oracle for SHA-256 would require a reversible
quantum circuit for SHA-256 plus ancilla management.

Candidate serialization is explicit and consistent:
    candidate_bytes = header || nonce.to_bytes(nonce_bytes, "big")
    digest          = SHA256(candidate_bytes)

The full 256-bit SHA-256 digest is always calculated. `--match-bits` selects
the most-significant digest bits used by the *toy oracle predicate*. A full
256-bit equality predicate in the small N^2 candidate domain is supported by
--match-bits 256, but searching an actual cryptographic nonce range is not
made feasible by this simulator.

Use --leading-zeros K to search for digests whose first K bits are zero
(e.g., --leading-zeros 32 for 32 leading zero bits, like Bitcoin mining).
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


def mask_for_bits(bits: int) -> int:
    if not 1 <= bits <= 256:
        raise ValueError("bits must be in [1, 256]")
    return (1 << bits) - 1


def parse_hex_bytes(value: str) -> bytes:
    value = value.strip().removeprefix("0x").replace(" ", "")
    if len(value) % 2:
        value = "0" + value
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("header must be valid hexadecimal") from exc


def candidate_message(header: bytes, nonce: int, nonce_bytes: int) -> bytes:
    if nonce < 0 or nonce >= (1 << (8 * nonce_bytes)):
        raise ValueError(f"nonce must fit in {nonce_bytes} byte(s)")
    return header + nonce.to_bytes(nonce_bytes, "big")


def sha256_digest(header: bytes, nonce: int, nonce_bytes: int) -> bytes:
    return hashlib.sha256(candidate_message(header, nonce, nonce_bytes)).digest()


def digest_prefix_int(digest: bytes, match_bits: int) -> int:
    """Extract the most-significant `match_bits` from a 256-bit digest."""
    full = int.from_bytes(digest, "big")
    if match_bits <= 0:
        return 0
    if match_bits >= 256:
        return full
    return full >> (256 - match_bits)


def sha256_compress(state: list[int], chunk: bytes) -> list[int]:
    """Reference one-block SHA-256 compression routine for validation only."""
    if len(chunk) != 64:
        raise ValueError("sha256_compress requires exactly one 64-byte chunk")
    K = (
        0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
        0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
        0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
        0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
        0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
        0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
        0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
        0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2,
    )
    m = 0xFFFFFFFF
    def ror(x: int, n: int) -> int: return ((x >> n) | (x << (32 - n))) & m
    def ch(x: int, y: int, z: int) -> int: return (x & y) ^ ((~x) & z)
    def maj(x: int, y: int, z: int) -> int: return (x & y) ^ (x & z) ^ (y & z)
    w = [int.from_bytes(chunk[i:i+4], "big") for i in range(0, 64, 4)]
    for i in range(16, 64):
        s0 = ror(w[i-15], 7) ^ ror(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = ror(w[i-2], 17) ^ ror(w[i-2], 19) ^ (w[i-2] >> 10)
        w.append((w[i-16] + s0 + w[i-7] + s1) & m)
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)
        t1 = (h + s1 + ch(e, f, g) + K[i] + w[i]) & m
        s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)
        t2 = (s0 + maj(a, b, c)) & m
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & m, c, b, a, (t1 + t2) & m
    return [
        (state[0] + a) & m, (state[1] + b) & m, (state[2] + c) & m, (state[3] + d) & m,
        (state[4] + e) & m, (state[5] + f) & m, (state[6] + g) & m, (state[7] + h) & m,
    ]


def pure_sha256_one_block(data: bytes) -> bytes:
    """Reference SHA-256 for messages shorter than 56 bytes."""
    if len(data) >= 56:
        raise ValueError("reference routine accepts messages shorter than 56 bytes")
    padded = data + b"\x80" + b"\x00" * (55 - len(data)) + (8 * len(data)).to_bytes(8, "big")
    initial = [0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19]
    return b"".join(x.to_bytes(4, "big") for x in sha256_compress(initial, padded))


def make_equal_amplitude_field(N: int) -> np.ndarray:
    return np.full((N, N), 1.0 / N, dtype=np.complex128)


def field_norm(field: np.ndarray) -> float:
    return float(np.vdot(field, field).real)


def diffusion_2d(field: np.ndarray) -> np.ndarray:
    return 2.0 * np.mean(field) - field


def recommended_iterations(candidate_count: int, marked_count: int) -> int:
    theta = math.asin(math.sqrt(marked_count / candidate_count))
    return max(0, int(round(math.pi / (4.0 * theta) - 0.5)))


def build_oracle_table(N: int, header: bytes, nonce_start: int, nonce_bytes: int,
                       match_bits: int, target_prefix: int | None,
                       leading_zeros: int | None) -> dict:
    count = N * N
    nonces = np.arange(nonce_start, nonce_start + count, dtype=np.uint64)
    if int(nonces[-1]) >= (1 << (8 * nonce_bytes)):
        raise ValueError("candidate range exceeds the selected nonce width")
    digests = [sha256_digest(header, int(nonce), nonce_bytes) for nonce in nonces]

    if leading_zeros is not None:
        if leading_zeros < 1 or leading_zeros > 256:
            raise ValueError("--leading-zeros must be in [1, 256]")
        target_prefix = 0
        match_bits = leading_zeros

    if target_prefix is None:
        prefixes = np.array([digest_prefix_int(d, match_bits) for d in digests], dtype=object)
        target_prefix = int(prefixes[0])
    else:
        prefixes = np.array([digest_prefix_int(d, match_bits) for d in digests], dtype=object)
    target_prefix &= mask_for_bits(match_bits)
    matches = np.equal(prefixes, target_prefix)
    return {
        "nonces": nonces,
        "digests": digests,
        "prefixes": prefixes,
        "target_prefix": target_prefix,
        "matches": matches,
        "match_bits": match_bits,
    }


def optical_grover(N: int, table: dict, iterations: int | None) -> dict:
    matches = table["matches"]
    marked_count = int(matches.sum())
    if marked_count == 0:
        raise RuntimeError("No candidate matches the oracle predicate in this N^2 window.")
    if iterations is None:
        iterations = recommended_iterations(N * N, marked_count)
    phase = np.where(matches, -1.0 + 0.0j, 1.0 + 0.0j).reshape(N, N)
    state = make_equal_amplitude_field(N)
    norm0 = field_norm(state)
    history = []
    for _ in range(iterations):
        state = diffusion_2d(state * phase)
        flat = state.reshape(-1)
        history.append(float(np.vdot(flat[matches], flat[matches]).real))
    p = np.abs(state) ** 2
    best_idx = int(np.argmax(p.reshape(-1)))
    return {
        "state": state, "probability": p, "phase": phase, "history": history,
        "iterations": iterations, "marked_count": marked_count, "best_idx": best_idx,
        "initial_norm": norm0, "final_norm": field_norm(state),
    }


def bit_image(values: list[int] | np.ndarray, width: int, max_rows: int = 2048) -> np.ndarray:
    values = list(values)
    if len(values) > max_rows:
        step = math.ceil(len(values) / max_rows)
        values = values[::step]
    out = np.zeros((len(values), width), dtype=np.uint8)
    for row, value in enumerate(values):
        for bit in range(width):
            out[row, width - 1 - bit] = (int(value) >> bit) & 1
    return out


def plot_matrix(ax, data: np.ndarray, title: str, cmap: str = "viridis") -> None:
    ax.imshow(data, origin="lower", interpolation="nearest", aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")


def make_visualization(N: int, header: bytes, nonce_bytes: int, match_bits: int,
                       table: dict, result: dict, out_path: str) -> None:
    nonces, digests, matches = table["nonces"], table["digests"], table["matches"]
    probability = result["probability"]
    candidate_count = N * N
    digest_ints = [int.from_bytes(d, "big") for d in digests]
    target = table["target_prefix"]
    xor = np.array([int(prefix) ^ target for prefix in table["prefixes"]], dtype=object)
    xor_log = np.array([math.log2(int(x) + 1) for x in xor], dtype=float).reshape(N, N)

    nonce_width = 8 * nonce_bytes
    nonce_bits = bit_image(nonces, nonce_width)
    digest_bits = bit_image(digest_ints, 256)

    fig = plt.figure(figsize=(22, 15), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.35, 0.75))
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(4)] for r in range(3)])

    plot_matrix(axes[0, 0], np.ones((N, N)), f"{N} x {N} UNIFORM AMPLITUDE", "Greys")
    plot_matrix(axes[0, 1], matches.reshape(N, N).astype(float), "ORACLE MARKS", "Greys")
    plot_matrix(axes[0, 2], result["phase"].real, "PHASE ORACLE (+1 / -1)", "coolwarm")
    plot_matrix(axes[0, 3], probability, "FINAL PROBABILITY", "inferno")

    plot_matrix(axes[1, 0], nonce_bits, f"NONCE BITS (sampled; {nonce_width} bits)", "Greys")
    axes[1, 0].set_xlabel("nonce bit (MSB to LSB)")
    axes[1, 0].set_ylabel("sampled candidate")
    plot_matrix(axes[1, 1], digest_bits, "FULL SHA-256 DIGEST BITS (sampled; 256 bits)", "Greys")
    axes[1, 1].set_xlabel("digest bit (MSB to LSB)")
    axes[1, 1].set_ylabel("sampled candidate")
    plot_matrix(axes[1, 2], xor_log, f"PREFIX XOR TARGET ({match_bits} bits), log2(x+1)", "magma")
    ax = axes[1, 3]
    ax.plot(np.arange(1, len(result["history"]) + 1), result["history"], marker="o")
    ax.set_title("MARKED-SET PROBABILITY")
    ax.set_xlabel("Grover iteration")
    ax.set_ylabel("P(all marked candidates)")
    ax.grid(alpha=0.25)

    for ax in axes[2]:
        ax.axis("off")
    best = result["best_idx"]
    marked_nonces = [int(x) for x in nonces[matches]]
    best_nonce = int(nonces[best])
    best_digest = digests[best].hex()
    axes[2, 0].text(0.01, 0.98, "\n".join([
        "MODEL", "", f"Plane: {N} x {N}", f"Candidates: {candidate_count:,}",
        f"Initial amplitude: 1/{N}", f"Initial norm: {result['initial_norm']:.12f}",
        f"Final norm: {result['final_norm']:.12f}", f"Rounds: {result['iterations']}",
    ]), va="top", family="monospace", fontsize=11)
    axes[2, 1].text(0.01, 0.98, "\n".join([
        "SHA-256 SERIALIZATION", "", f"Header bytes: {len(header)}", f"Nonce bytes: {nonce_bytes}",
        "Message = header || nonce", "Nonce byte order = big-endian", "", f"Oracle prefix width: {match_bits} bits",
    ]), va="top", family="monospace", fontsize=11)
    axes[2, 2].text(0.01, 0.98, "\n".join([
        "ORACLE RESULT", "", f"Target prefix: 0x{target:X}", f"Marked count: {result['marked_count']}",
        f"Marked nonces: {marked_nonces[:6]}" + (" ..." if len(marked_nonces) > 6 else ""),
        "", "Full-digest equality requires", "--match-bits 256.",
    ]), va="top", family="monospace", fontsize=11)
    axes[2, 3].text(0.01, 0.98, "\n".join([
        "MEASUREMENT", "", f"Best nonce: {best_nonce}", f"Position: ({best // N}, {best % N})",
        f"P(best): {probability.reshape(-1)[best]:.8f}", f"P(marked): {probability.reshape(-1)[matches].sum():.8f}",
        "", f"SHA256(best): {best_digest[:32]}", best_digest[32:],
    ]), va="top", family="monospace", fontsize=9)

    fig.suptitle("REAL SHA-256 CANDIDATE ORACLE + CLASSICAL GROVER-STYLE AMPLIFICATION", fontsize=16)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def verify_reference(header: bytes, nonce: int, nonce_bytes: int) -> bool:
    data = candidate_message(header, nonce, nonce_bytes)
    if len(data) >= 56:
        return False
    return pure_sha256_one_block(data) == hashlib.sha256(data).digest()


def run(N: int, header: bytes, nonce_start: int, nonce_bytes: int,
        match_bits: int, target_prefix: int | None, leading_zeros: int | None,
        iterations: int | None, out_path: str) -> dict:
    if N < 2:
        raise ValueError("N must be >= 2")
    if nonce_bytes < 1 or nonce_bytes > 8:
        raise ValueError("nonce_bytes must be in [1, 8]")
    if len(header) + nonce_bytes >= 56:
        raise ValueError("this reference implementation limits header + nonce to fewer than 56 bytes")
   

    table = build_oracle_table(N, header, nonce_start, nonce_bytes, match_bits, target_prefix, leading_zeros)
    result = optical_grover(N, table, iterations)
    best = result["best_idx"]
    hashes = [digest_prefix_int(d, table["match_bits"]) for d in table["digests"]]
    stats = Counter(hashes)

    print("=" * 78)
    print("REAL SHA-256 ORACLE / CLASSICAL GROVER-STYLE SIMULATION")
    print("=" * 78)
    print(f"Plane:                  {N} x {N}")
    print(f"Candidates:             {N*N:,}")
    print(f"Header (hex):           {header.hex() or '<empty>'}")
    print(f"Nonce range:            [{nonce_start}, {nonce_start + N*N - 1}]")
    print(f"Nonce serialization:    {nonce_bytes} byte(s), big-endian")
    if leading_zeros is not None:
        print(f"Match predicate:        first {leading_zeros} SHA-256 bits = 0 (leading zeros mode)")
    else:
        print(f"Match predicate:        first {match_bits} SHA-256 bits")
    print(f"Target prefix:          0x{table['target_prefix']:X}")
    print(f"Marked candidates:      {result['marked_count']}")
    print(f"Grover rounds:          {result['iterations']}")
    print(f"Initial / final norm:   {result['initial_norm']:.12f} / {result['final_norm']:.12f}")
    print(f"Best nonce:             {int(table['nonces'][best])}")
    print(f"Best digest:            {table['digests'][best].hex()}")
    print(f"P(best):                {result['probability'].reshape(-1)[best]:.8f}")
    print(f"P(all marked):          {result['probability'].reshape(-1)[table['matches']].sum():.8f}")
    print(f"Distinct prefixes:      {len(stats)}")
    print(f"Largest prefix bucket:  {max(stats.values())}")
    print(f"Pure-Python SHA match:  {verify_reference(header, int(table['nonces'][0]), nonce_bytes)}")
    print(f"PNG:                    {out_path}")

    make_visualization(N, header, nonce_bytes, table["match_bits"], table, result, out_path)
    return {"table": table, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Real SHA-256 oracle plus classical Grover-style simulator.")
    parser.add_argument("--size", type=int, default=8, help="N for the N x N candidate plane; use 256 for 65,536 candidates.")
    parser.add_argument("--header-hex", type=parse_hex_bytes, default=b"", help="Fixed header/prefix bytes as hex.")
    parser.add_argument("--nonce-start", type=lambda x: int(x, 0), default=0, help="First integer nonce in the candidate window.")
    parser.add_argument("--nonce-bytes", type=int, default=4, help="Fixed nonce serialization width in bytes.")
    parser.add_argument("--match-bits", type=int, default=None, help="Most-significant SHA-256 bits compared by the oracle; default is min(N, 256).")
    parser.add_argument("--target-prefix", type=lambda x: int(x, 0), default=None, help="Oracle target prefix; default uses the first candidate's prefix.")
    parser.add_argument("--leading-zeros", type=int, default=None, help="Set target prefix to zero for the first N bits (e.g. --leading-zeros 32 for 32 leading zero bits). Overrides --target-prefix if provided.")
    parser.add_argument("--iterations", type=int, default=None, help="Override the recommended amplitude-amplification count.")
    parser.add_argument("--output", default=None, help="Output PNG filename.")
    args = parser.parse_args()
    match_bits = args.match_bits if args.match_bits is not None else min(args.size, 256)
    if match_bits > 256:
        match_bits = 256
    out = args.output or f"sha256_oracle_{args.size}x{args.size}.png"
    run(args.size, args.header_hex, args.nonce_start, args.nonce_bytes,
        match_bits, args.target_prefix, args.leading_zeros, args.iterations, out)


if __name__ == "__main__":
    main()
