"""
nonce_search_comparison.py

Compares two ways of brute-forcing a SHA-256 nonce search
("mining" for a hash below a difficulty target):

  1. YOUR VERSION   - builds a hex string (base + zero-padding + nonce)
                      and hashes the resulting bytes, matching the
                      structure of your original script (bugs fixed:
                      single loop over the nonce, f-string hashing,
                      and digest compared as hex, not raw bytes).

  2. REGULAR VERSION - the standard/idiomatic way: nonce as raw
                       big-endian bytes appended to a base payload,
                       hashed directly. This is closer to how real
                       PoW nonce search is written (e.g. Bitcoin
                       Core's mining loop, conceptually).

Both are timed on the SAME target difficulty (leading zero BITS,
not bytes, so the difficulty knob is fine-grained) so hashrate and
attempt counts are directly comparable.

Nothing here searches a real Bitcoin block header or connects to
the network - it's a local demonstration of nonce-search mechanics
and CPU hashrate, using hashlib exactly as your script did.
"""

import hashlib
import time


# ======================================================================
# SHARED: difficulty check
# ======================================================================

def leading_zero_bits(digest: bytes) -> int:
    """Count leading zero BITS in a digest (finer-grained than hex nibbles)."""
    count = 0
    for byte in digest:
        if byte == 0:
            count += 8
            continue
        # count leading zero bits within this nonzero byte
        count += 8 - byte.bit_length()
        break
    return count


def meets_difficulty(digest: bytes, target_bits: int) -> bool:
    return leading_zero_bits(digest) >= target_bits


# ======================================================================
# 1) YOUR VERSION  (fixed: hex-string building, same spirit as original)
# ======================================================================

def make_padded_hex_with_counter(base_hex: str, counter: int, total_width: int = 64) -> str:
    """
    Hex string = left-padded with 0s + base_hex + counter (as hex pairs).
    (Unchanged from your original.)
    """
    counter_hex = format(counter, "x")
    if len(counter_hex) % 2 == 1:
        counter_hex = "0" + counter_hex
    combined = base_hex + counter_hex
    if len(combined) > total_width:
        return combined
    return combined.rjust(total_width, "0")


def your_version_search(base_hex: str, target_bits: int, max_counter: int, total_width: int = 64):
    """
    Fixed version of your bruteforce_nonce_with_hash:
      - single loop over the nonce (no bogus nested c/x loop)
      - hashes the ACTUAL bytes built from the hex string
      - compares leading-zero-BITS against a difficulty target
        (a real target check, not an exact digest match, since
        exact-digest matching against an unknown target is not
        how nonce search works)
    """
    attempts = 0
    start = time.perf_counter()

    for counter in range(max_counter):
        hex_str = make_padded_hex_with_counter(base_hex, counter, total_width)
        data = bytes.fromhex(hex_str)
        digest = hashlib.sha256(data).digest()
        attempts += 1

        if meets_difficulty(digest, target_bits):
            elapsed = time.perf_counter() - start
            return {
                "found": True,
                "nonce": counter,
                "hex_str": hex_str,
                "hash_hex": digest.hex(),
                "attempts": attempts,
                "elapsed": elapsed,
            }

    elapsed = time.perf_counter() - start
    return {"found": False, "attempts": attempts, "elapsed": elapsed}


# ======================================================================
# 2) REGULAR VERSION (idiomatic nonce-as-bytes mining loop)
# ======================================================================

def regular_version_search(base: bytes, target_bits: int, max_counter: int):
    """
    Standard approach: nonce as 4-byte big-endian integer appended
    directly to the base payload, no hex round-tripping.
    """
    attempts = 0
    start = time.perf_counter()

    for nonce in range(max_counter):
        data = base + nonce.to_bytes(4, "big")
        digest = hashlib.sha256(data).digest()
        attempts += 1

        if meets_difficulty(digest, target_bits):
            elapsed = time.perf_counter() - start
            return {
                "found": True,
                "nonce": nonce,
                "data_hex": data.hex(),
                "hash_hex": digest.hex(),
                "attempts": attempts,
                "elapsed": elapsed,
            }

    elapsed = time.perf_counter() - start
    return {"found": False, "attempts": attempts, "elapsed": elapsed}


# ======================================================================
# BENCHMARK
# ======================================================================

def run_comparison(target_bits: int = 20, max_counter: int = 20_000_000):
    print("=" * 70)
    print(f"NONCE SEARCH COMPARISON  (target: {target_bits} leading zero bits)")
    print(f"Expected average attempts for this difficulty: ~{2**target_bits:,}")
    print("=" * 70)

    print("\n--- YOUR VERSION (hex-padding approach) ---")
    result_a = your_version_search(base_hex="", target_bits=target_bits, max_counter=max_counter)
    print_result(result_a)

    print("\n--- REGULAR VERSION (nonce-as-bytes approach) ---")
    result_b = regular_version_search(base=b"", target_bits=target_bits, max_counter=max_counter)
    print_result(result_b)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if result_a["found"] and result_b["found"]:
        rate_a = result_a["attempts"] / result_a["elapsed"]
        rate_b = result_b["attempts"] / result_b["elapsed"]
        print(f"Your version    : {rate_a:,.0f} H/s  ({result_a['attempts']:,} attempts, {result_a['elapsed']:.4f}s)")
        print(f"Regular version : {rate_b:,.0f} H/s  ({result_b['attempts']:,} attempts, {result_b['elapsed']:.4f}s)")
        faster = "Regular version" if rate_b > rate_a else "Your version"
        ratio = max(rate_a, rate_b) / min(rate_a, rate_b)
        print(f"\n{faster} is ~{ratio:.2f}x faster.")
        print("(Difference comes from hex-encode/decode overhead in your version:")
        print(" format() -> rjust() -> bytes.fromhex() per attempt, vs a direct")
        print(" int.to_bytes() concatenation in the regular version. Same SHA-256")
        print(" cost either way - the gap is pure Python string/bytes overhead.)")


def print_result(result: dict):
    if result["found"]:
        rate = result["attempts"] / result["elapsed"]
        print(f"  Found nonce   : {result['nonce']:,}")
        print(f"  Hash          : {result['hash_hex']}")
        print(f"  Attempts      : {result['attempts']:,}")
        print(f"  Elapsed       : {result['elapsed']:.4f} s")
        print(f"  Hashrate      : {rate:,.0f} H/s")
    else:
        print(f"  Not found within {result['attempts']:,} attempts ({result['elapsed']:.4f}s).")
        print("  Increase max_counter or lower target_bits.")


if __name__ == "__main__":
    # 20 bits ~= 1 in 1,048,576 odds per attempt - fast enough to run live,
    # slow enough to give a stable hashrate measurement.
    run_comparison(target_bits=24, max_counter=20_000_000_000)
