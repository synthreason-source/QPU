#!/usr/bin/env python3
"""
bitstring_walker.py

Address-by-address bitstring manipulation over the spatial plane,
using the same addressing model as QPU.py:

    address = temporal_bin * spatial_modes + spatial_mode

NO array of size (temporal_bins * spatial_modes) is ever allocated.
Each address is visited one at a time, its bitstring is built,
manipulated, and (optionally) discarded/written out immediately.

For each address this script computes:

    address_bits    full N-bit binary string of the combined address
    spatial_bits    low-order bits -> spatial_mode
    temporal_bits   high-order bits -> temporal_bin
    popcount        number of set bits (Hamming weight)
    parity          popcount % 2  (0 = even, 1 = odd)
    gray_code       binary-reflected Gray code of the address
    bit_reversed    address_bits read backwards, reinterpreted as int
    xor_masked      address XOR'd with a rolling/user mask
    is_prime        whether the combined decimal address is prime

Usage examples
--------------
    # default: 32x32 spatial plane, 64 temporal bins (matches QPU.py docstring)
    python bitstring_walker.py

    # smaller plane, print every address
    python bitstring_walker.py --modes-x 4 --modes-y 4 --modes-z 2 --show all

    # only show addresses whose bitstring is a palindrome
    python bitstring_walker.py --filter palindrome

    # apply a fixed XOR mask (hex) to every address's bitstring
    python bitstring_walker.py --xor-mask 0x00FF
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# ==============================================================
# PRIME TEST (same as QPU.py)
# ==============================================================

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


# ==============================================================
# ADDRESS <-> BITSTRING HELPERS
# ==============================================================

def optical_address(temporal_bin: int, spatial_mode: int, spatial_modes: int) -> int:
    """Same flattening rule as QPU.py."""
    return int(temporal_bin) * int(spatial_modes) + int(spatial_mode)


def to_bits(value: int, width: int) -> str:
    return format(value, f"0{width}b")


def popcount(bits: str) -> int:
    return bits.count("1")


def parity(bits: str) -> int:
    return popcount(bits) & 1


def to_gray(value: int) -> int:
    """Binary-reflected Gray code: g = v XOR (v >> 1)."""
    return value ^ (value >> 1)


def bit_reverse(bits: str) -> int:
    return int(bits[::-1], 2)


def is_palindrome(bits: str) -> bool:
    return bits == bits[::-1]


# ==============================================================
# ADDRESS-BY-ADDRESS BITSTRING MANIPULATOR
# ==============================================================

class BitstringWalker:
    """
    Walks the temporal x spatial optical address space one address
    at a time, applying bitstring manipulations. Mirrors the
    "acquire -> process -> discard" streaming pattern in QPU.py's
    TemporalOpticalEngine, but for pure address/bitstring math
    (no camera, no ITO, no numpy array of the full space).
    """

    def __init__(self, rows: int, cols: int, temporal_bins: int, xor_mask: int = 0):
        self.rows = int(rows)
        self.cols = int(cols)
        self.spatial_modes = self.rows * self.cols
        self.temporal_bins = int(temporal_bins)
        self.total_modes = self.spatial_modes * self.temporal_bins

        self.qubits = max(1, (self.total_modes - 1).bit_length())
        self.spatial_bits_width = max(1, (self.spatial_modes - 1).bit_length())
        self.temporal_bits_width = self.qubits - self.spatial_bits_width

        self.xor_mask = int(xor_mask) & ((1 << self.qubits) - 1)

    def spatial_rc(self, spatial_mode: int) -> tuple[int, int]:
        """Row/col within the spatial plane for a given spatial_mode index."""
        return divmod(spatial_mode, self.cols)

    def manipulate(self, temporal_bin: int, spatial_mode: int) -> dict:
        """
        Build and manipulate the bitstring for exactly ONE address.
        This is the unit of work that gets called address by address.
        """
        address = optical_address(temporal_bin, spatial_mode, self.spatial_modes)
        address_bits = to_bits(address, self.qubits)

        # split combined bitstring into temporal/spatial fields
        temporal_field = address_bits[: self.temporal_bits_width] or "0"
        spatial_field = address_bits[self.temporal_bits_width :]

        xor_value = address ^ self.xor_mask
        r, c = self.spatial_rc(spatial_mode)

        return {
            "temporal_bin": temporal_bin,
            "spatial_mode": spatial_mode,
            "row": r,
            "col": c,
            "address": address,
            "address_bits": address_bits,
            "temporal_field": temporal_field,
            "spatial_field": spatial_field,
            "popcount": popcount(address_bits),
            "parity": parity(address_bits),
            "gray_code": to_bits(to_gray(address), self.qubits),
            "bit_reversed": bit_reverse(address_bits),
            "xor_masked": to_bits(xor_value, self.qubits),
            "is_palindrome": is_palindrome(address_bits),
            "is_prime": is_prime(address),
        }

    def walk(self, filter_fn=None):
        """
        Generator: yields one manipulated-address record at a time,
        address by address, spatial plane first (row-major) within
        each temporal bin. Nothing beyond the current record is
        held in memory.
        """
        for t in range(self.temporal_bins):
            for s in range(self.spatial_modes):
                record = self.manipulate(t, s)
                if filter_fn is None or filter_fn(record):
                    yield record


# ==============================================================
# FILTERS
# ==============================================================

FILTERS = {
    "all": None,
    "prime": lambda rec: rec["is_prime"],
    "palindrome": lambda rec: rec["is_palindrome"],
    "even-parity": lambda rec: rec["parity"] == 0,
    "odd-parity": lambda rec: rec["parity"] == 1,
}


# ==============================================================
# CLI
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Address-by-address bitstring manipulation over the spatial plane."
    )

    parser.add_argument("--modes-x", type=int, default=32, help="Spatial columns.")
    parser.add_argument("--modes-y", type=int, default=32, help="Spatial rows.")
    parser.add_argument("--modes-z", type=int, default=64, help="Temporal bins.")

    parser.add_argument(
        "--xor-mask",
        type=lambda x: int(x, 0),
        default=0,
        help="Integer (accepts 0x.. hex) XOR mask applied to every address bitstring.",
    )

    parser.add_argument(
        "--filter",
        choices=list(FILTERS.keys()),
        default="prime",
        help="Which addresses to keep while walking (default: prime, matches QPU.py's --prime-only spirit).",
    )

    parser.add_argument(
        "--show",
        choices=["all", "head"],
        default="head",
        help="'head' prints only the first N matches (see --limit); 'all' prints every match.",
    )

    parser.add_argument("--limit", type=int, default=32, help="Max rows to print when --show head.")

    parser.add_argument("--output", default=None, help="Optional CSV path to save every matched address.")

    return parser.parse_args()


def main():
    args = parse_args()

    walker = BitstringWalker(
        rows=args.modes_y,
        cols=args.modes_x,
        temporal_bins=args.modes_z,
        xor_mask=args.xor_mask,
    )

    print("=" * 100)
    print("BITSTRING WALKER — ADDRESS BY ADDRESS")
    print("=" * 100)
    print(f"Spatial plane      : {walker.rows} x {walker.cols}  ({walker.spatial_modes:,} modes)")
    print(f"Temporal bins      : {walker.temporal_bins:,}")
    print(f"Total addresses    : {walker.total_modes:,}")
    print(f"Address width      : {walker.qubits} bits "
          f"({walker.temporal_bits_width} temporal + {walker.spatial_bits_width} spatial)")
    print(f"XOR mask           : {to_bits(walker.xor_mask, walker.qubits)}")
    print(f"Filter             : {args.filter}")
    print()

    filter_fn = FILTERS[args.filter]

    fieldnames = [
        "address", "address_bits", "temporal_bin", "spatial_mode", "row", "col",
        "temporal_field", "spatial_field", "popcount", "parity",
        "gray_code", "bit_reversed", "xor_masked", "is_palindrome", "is_prime",
    ]

    writer = None
    out_file = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

    printed = 0
    matched = 0

    try:
        for record in walker.walk(filter_fn):
            matched += 1

            if writer is not None:
                writer.writerow(record)

            if args.show == "all" or printed < args.limit:
                print(
                    f"addr={record['address']:>7}  "
                    f"bits={record['address_bits']}  "
                    f"[T:{record['temporal_field']}|S:{record['spatial_field']}]  "
                    f"t={record['temporal_bin']:>4} (r{record['row']},c{record['col']})  "
                    f"pop={record['popcount']:>2}  par={record['parity']}  "
                    f"gray={record['gray_code']}  xor={record['xor_masked']}"
                )
                printed += 1

    finally:
        if out_file is not None:
            out_file.close()

    print()
    print(f"Matched addresses  : {matched:,} / {walker.total_modes:,}")
    if args.output:
        print(f"Saved to           : {args.output}")


if __name__ == "__main__":
    main()
